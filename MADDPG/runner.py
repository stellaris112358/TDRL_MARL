from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境下使用非交互式后端
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Performance-related knobs. Defaults preserve current behavior when possible,
        # while letting us reduce update and I/O overhead from config.
        self.train_interval = max(1, int(getattr(args, 'train_interval', 1)))
        self.updates_per_train = max(1, int(getattr(args, 'updates_per_train', 1)))
        self.loss_log_interval = max(1, int(getattr(args, 'loss_log_interval', 100)))
        self.explore_log_interval = max(1, int(getattr(args, 'explore_log_interval', 500)))
        self.eval_save_interval = max(1, int(getattr(args, 'eval_save_interval', 1)))
        self.eval_plot_interval = max(1, int(getattr(args, 'eval_plot_interval', 1)))
        self.update_counter = 0

        # TensorBoard writer：日志写入 <save_path>/tb_logs/
        tb_log_dir = os.path.join(self.save_path, 'tb_logs')
        self.writer = SummaryWriter(log_dir=tb_log_dir)
        print(f'TensorBoard logs -> {tb_log_dir}')
        print(f'  tensorboard --logdir {tb_log_dir}')

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def _should_train_this_step(self, time_step):
        return (
            self.buffer.current_size >= self.args.batch_size
            and (time_step + 1) % self.train_interval == 0
        )

    def _run_maddpg_updates(self, time_step, recent_actor_loss, recent_critic_loss):
        for _ in range(self.updates_per_train):
            transitions = self.buffer.sample(self.args.batch_size)
            self.update_counter += 1
            for idx, agent in enumerate(self.agents):
                other_agents = self.agents.copy()
                other_agents.remove(agent)
                a_loss, c_loss = agent.learn(transitions, other_agents)
                recent_actor_loss[idx] = a_loss
                recent_critic_loss[idx] = c_loss
                if self.update_counter % self.loss_log_interval == 0:
                    self.writer.add_scalar(f'loss/actor_agent{idx}', a_loss, self.update_counter)
                    self.writer.add_scalar(f'loss/critic_agent{idx}', c_loss, self.update_counter)

    def _save_returns(self, returns):
        np.save(self.save_path + '/returns.pkl', returns)

    def _save_eval_plot(self, returns):
        plt.figure()
        plt.plot(range(len(returns)), returns)
        plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/plt.png', format='png')
        plt.close()

    def run(self):
        returns = []
        # 用于 tqdm postfix 的滑动窗口统计
        recent_actor_loss  = [0.0] * self.args.n_agents
        recent_critic_loss = [0.0] * self.args.n_agents
        recent_reward = 0.0

        pbar = tqdm(
            range(self.args.time_steps),
            desc='Training',
            unit='step',
            dynamic_ncols=True,
        )
        for time_step in pbar:
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
                ep_rewards = np.zeros(self.args.n_agents)

            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next

            # 累计本 episode 奖励
            for i in range(self.args.n_agents):
                ep_rewards[i] += r[i]

            if self._should_train_this_step(time_step):
                self._run_maddpg_updates(time_step, recent_actor_loss, recent_critic_loss)

            # episode 结束时记录本 episode 奖励
            if (time_step + 1) % self.episode_limit == 0:
                ep_mean = float(np.mean(ep_rewards))
                recent_reward = ep_mean
                episode_id = (time_step + 1) // self.episode_limit
                self.writer.add_scalar('reward/mean_episode_return', ep_mean, time_step)
                for i in range(self.args.n_agents):
                    self.writer.add_scalar(f'reward/agent{i}_episode_return', float(ep_rewards[i]), time_step)

            # 更新 tqdm postfix（每 50 步刷新一次，避免刷新过频）
            if time_step % 50 == 0:
                postfix = {
                    'rew':  f'{recent_reward:.2f}',
                    'a_loss': f'{np.mean(recent_actor_loss):.4f}',
                    'c_loss': f'{np.mean(recent_critic_loss):.4f}',
                    'noise':   f'{self.noise:.3f}',
                    'eps':     f'{self.epsilon:.3f}',
                }
                pbar.set_postfix(postfix)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                eval_return = self.evaluate()
                returns.append(eval_return)
                eval_episode = time_step // self.episode_limit
                self.writer.add_scalar('eval/mean_return', eval_return, eval_episode)
                if len(returns) % self.eval_save_interval == 0:
                    self._save_returns(returns)
                if len(returns) % self.eval_plot_interval == 0:
                    self._save_eval_plot(returns)

            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)

            # TensorBoard: 探索参数
            if time_step % self.explore_log_interval == 0:
                self.writer.add_scalar('explore/noise',   self.noise,   time_step)
                self.writer.add_scalar('explore/epsilon', self.epsilon, time_step)

        self.writer.close()
        self._save_returns(returns)
        if returns:
            self._save_eval_plot(returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if self.args.evaluate:
                    self.env.render()
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            # print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
