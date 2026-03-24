"""
train_tdrl.py — MADDPG + TdRL 训练入口

用法：
  # 默认配置
  python train_tdrl.py

  # 覆盖参数
  python train_tdrl.py time_steps=1000000 tdrl.reward_update=100

  # 评估模式
  python train_tdrl.py evaluate=true
"""

import os
import sys
from argparse import Namespace

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# 确保 MADDPG 目录在 sys.path 最前面（runner, agent, maddpg 等都在此）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 上级目录（TDRL_MARL 根目录）追加到末尾，以便导入 tester / reward_model
# 但不能在前面，否则根目录的 agent/ 包会覆盖 MADDPG 下的 agent.py
_REPO_ROOT = os.path.dirname(_PROJECT_ROOT)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from runner import Runner
from common.pettingzoo_wrapper import make_pz_env
from tester import TestDict
from reward_model import MATdRLRewardModel


# ── helper ────────────────────────────────────────────────

def cfg_to_namespace(cfg: DictConfig) -> Namespace:
    """将 Hydra DictConfig 展平为 argparse.Namespace，兼容原有 args.xxx 调用。"""
    raw = OmegaConf.to_container(cfg, resolve=True)
    flat: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict) and k != "tdrl":
            flat.update(v)
        else:
            flat[k] = v
    return Namespace(**flat)


# ── TdRL-augmented Runner ────────────────────────────────

class TdRLRunner(Runner):
    """
    在原 Runner 基础上增加:
      1. 每步把数据喂给 MATdRLRewardModel.add_data
      2. 定期训练 reward model (return network + per-agent reward network)
      3. 用学到的 reward 替换 env reward 存入 buffer (可选 relabel)
    """

    def __init__(self, args, env, reward_model: MATdRLRewardModel):
        super().__init__(args, env)
        self.reward_model = reward_model
        self.tdrl_cfg = args.tdrl
        self._episode_count = 0

    # ── override run ──────────────────────────────────────

    def run(self):
        from tqdm import tqdm
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        returns = []
        recent_actor_loss  = [0.0] * self.args.n_agents
        recent_critic_loss = [0.0] * self.args.n_agents
        recent_reward = 0.0
        reward_model_ready = False

        pbar = tqdm(
            range(self.args.time_steps),
            desc='TdRL Training',
            unit='step',
            dynamic_ncols=True,
        )

        for time_step in pbar:
            # ── episode reset ──
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
                ep_rewards = np.zeros(self.args.n_agents)
                if time_step > 0:
                    self._episode_count += 1

            # ── select actions ──
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

            # ── env step ──
            s_next, r, done, info = self.env.step(actions)

            # ── feed data to reward model ──
            episode_done = done or ((time_step + 1) % self.episode_limit == 0)
            self.reward_model.add_data(
                obs_list=s[:self.args.n_agents],
                act_list=u,
                next_obs_list=s_next[:self.args.n_agents],
                rew_list=r[:self.args.n_agents],
                done=episode_done,
            )

            # ── choose which rewards to store ──
            if reward_model_ready and self.tdrl_cfg.get("relabel_buffer", True):
                # build joint_sa for reward prediction
                joint_sa = np.concatenate([
                    np.concatenate([s[i], u[i]]) for i in range(self.args.n_agents)
                ])
                r_pred = [
                    self.reward_model.r_hat(agent_id, joint_sa)
                    for agent_id in range(self.args.n_agents)
                ]
            else:
                r_pred = r[:self.args.n_agents]

            self.buffer.store_episode(
                s[:self.args.n_agents], u,
                r_pred,
                s_next[:self.args.n_agents],
            )
            s = s_next

            # ── accumulate ep rewards (env reward for logging) ──
            for i in range(self.args.n_agents):
                ep_rewards[i] += r[i]

            # ── MADDPG update ──
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for idx, agent in enumerate(self.agents):
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    a_loss, c_loss = agent.learn(transitions, other_agents)
                    recent_actor_loss[idx] = a_loss
                    recent_critic_loss[idx] = c_loss
                    self.writer.add_scalar(f'loss/actor_agent{idx}', a_loss, time_step)
                    self.writer.add_scalar(f'loss/critic_agent{idx}', c_loss, time_step)

            # ── episode end bookkeeping ──
            if (time_step + 1) % self.episode_limit == 0:
                ep_mean = float(np.mean(ep_rewards))
                recent_reward = ep_mean
                episode_id = (time_step + 1) // self.episode_limit
                self.writer.add_scalar('reward/mean_episode_return', ep_mean, episode_id)
                for i in range(self.args.n_agents):
                    self.writer.add_scalar(
                        f'reward/agent{i}_episode_return', float(ep_rewards[i]), episode_id)

            # ── TdRL reward model training ──
            if (time_step + 1) % self.episode_limit == 0:
                if time_step >= self.tdrl_cfg.get("num_interact", 5000):
                    if self._episode_count % self.tdrl_cfg.get("reward_update", 50) == 0:
                        self._train_reward_model(time_step)
                        reward_model_ready = True

            # ── tqdm postfix ──
            if time_step % 50 == 0:
                postfix = {
                    'rew': f'{recent_reward:.2f}',
                    'a_loss': f'{np.mean(recent_actor_loss):.4f}',
                    'c_loss': f'{np.mean(recent_critic_loss):.4f}',
                    'rm': 'ON' if reward_model_ready else 'OFF',
                }
                pbar.set_postfix(postfix)

            # ── evaluation ──
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                eval_return = self.evaluate()
                returns.append(eval_return)
                eval_episode = time_step // self.episode_limit
                self.writer.add_scalar('eval/mean_return', eval_return, eval_episode)
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
                plt.close()

            # ── exploration decay ──
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)

            if time_step % 500 == 0:
                self.writer.add_scalar('explore/noise', self.noise, time_step)
                self.writer.add_scalar('explore/epsilon', self.epsilon, time_step)

            np.save(self.save_path + '/returns.pkl', returns)

        self.writer.close()

    # ── reward model training subroutine ──────────────────

    def _train_reward_model(self, time_step):
        """One round of reward model training: prepare → train returns → train reward."""
        print(f"\n[TdRL] Training reward model @ step {time_step}")

        ind_mean, pf_mean = self.reward_model.prepare_training()
        self.writer.add_scalar('tdrl/pf_mean', float(np.mean(pf_mean)), time_step)
        for i, v in enumerate(ind_mean):
            self.writer.add_scalar(f'tdrl/ind_{i}', float(v), time_step)

        # train return network
        for epoch in range(self.tdrl_cfg.get("return_train_epochs", 5)):
            ret_acc, ret_loss = self.reward_model.train_returns()
        self.writer.add_scalar('tdrl/return_acc', float(np.mean(ret_acc)), time_step)
        self.writer.add_scalar('tdrl/return_loss', float(np.mean(ret_loss)), time_step)

        # train per-agent reward networks
        for epoch in range(self.tdrl_cfg.get("reward_train_epochs", 5)):
            agent_losses = self.reward_model.train_reward()
        for i, loss_val in agent_losses.items():
            self.writer.add_scalar(f'tdrl/reward_loss_agent{i}', loss_val, time_step)

        print(f"[TdRL] Return acc: {ret_acc}, Agent reward losses: {agent_losses}")

        # optionally relabel the entire buffer
        if self.tdrl_cfg.get("relabel_buffer", True):
            self._relabel_buffer()

    def _relabel_buffer(self):
        """用学到的 reward model 重标注 replay buffer 中已有的数据。"""
        n = self.buffer.current_size
        if n == 0:
            return

        for agent_id in range(self.args.n_agents):
            # 构造 joint_sa for all stored transitions
            all_obs = []
            all_act = []
            for i in range(self.args.n_agents):
                all_obs.append(self.buffer.buffer[f'o_{i}'][:n])
                all_act.append(self.buffer.buffer[f'u_{i}'][:n])
            # joint_sa: (n, sum(obs+act))
            joint_sa = np.concatenate(
                [np.concatenate([all_obs[i], all_act[i]], axis=-1)
                 for i in range(self.args.n_agents)],
                axis=-1,
            )
            r_pred = self.reward_model.r_hat_batch(agent_id, joint_sa)  # (n, 1)
            self.buffer.buffer[f'r_{agent_id}'][:n] = r_pred.squeeze(-1)


# ── main ──────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="conf", config_name="config_tdrl")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("MADDPG + TdRL 训练配置：")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    args = cfg_to_namespace(cfg)

    seed = getattr(args, "seed", args.tdrl.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env, args = make_pz_env(args)

    # ── build tester ──
    TesterClass = TestDict[args.scenario_name]
    tester = TesterClass()

    # ── build reward model ──
    reward_model = MATdRLRewardModel(
        n_agents=args.n_agents,
        obs_dims=args.obs_shape,
        act_dims=args.action_shape,
        device=device,
        tester=tester,
        use_kan=args.tdrl.get("use_kan", False),
        return_ensemble_size=args.tdrl.get("return_ensemble_size", 3),
        ensemble_size=args.tdrl.get("ensemble_size", 3),
        spectral_norm=args.tdrl.get("spectral_norm", False),
        loss_combine=args.tdrl.get("loss_combine", "early_stop"),
        change_penalty=args.tdrl.get("change_penalty", 0.0),
        mse_coef=args.tdrl.get("mse_coef", 1.0),
        es_coef=args.tdrl.get("es_coef", 10.0),
        lr=args.tdrl.get("lr", 3e-4),
        mb_size=args.tdrl.get("mb_size", 128),
        size_segment=args.tdrl.get("size_segment", 25),
        max_size=args.tdrl.get("max_size", 200),
    )

    # ── run ──
    runner = TdRLRunner(args, env, reward_model)

    if args.evaluate:
        returns = runner.evaluate()
        print(f"Average returns: {returns:.4f}")
    else:
        runner.run()


if __name__ == "__main__":
    main()
