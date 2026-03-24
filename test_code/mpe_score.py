# 使用训练好的 MADDPG actor 在 simple_spread_v3 上评估
import os
import sys
import warnings
from argparse import Namespace

import torch

# 先屏蔽第三方库的已知警告（需在导入前设置）
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="The environment `pettingzoo.mpe` has been moved to `mpe2`.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*", category=FutureWarning)

from pettingzoo.mpe import simple_spread_v3

# 保证可以导入 MADDPG 包（脚本位于 repo_root/test_code/）
repo_root = os.path.dirname(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


from MADDPG.maddpg.actor_critic import Actor


def build_args(env):
    agent_names = list(env.possible_agents)
    obs_shape = [env.observation_space(a).shape[0] for a in agent_names]
    action_shape = [env.action_space(a).shape[0] for a in agent_names]
    return Namespace(
        high_action=1,
        obs_shape=obs_shape,
        action_shape=action_shape,
    )


def load_actors(model_dir, args, device):
    actors = []
    for agent_id in range(len(args.obs_shape)):
        actor = Actor(args, agent_id).to(device)
        ckpt = os.path.join(model_dir, f"agent_{agent_id}", "actor_params.pkl")
        actor.load_state_dict(torch.load(ckpt, map_location=device))
        actor.eval()
        actors.append(actor)
    return actors


if __name__ == "__main__":
    env = simple_spread_v3.parallel_env(N=3, render_mode="rgb_array", dynamic_rescaling=False, continuous_actions=True)
    observations, infos = env.reset()
    agent_order = list(env.possible_agents)

    args = build_args(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型目录：MADDPG/model/simple_spread_v3/agent_x/actor_params.pkl
    repo_root = os.path.dirname(os.path.dirname(__file__))
    model_root = os.path.join(repo_root, "MADDPG", "model", "simple_spread_v3")
    actors = load_actors(model_root, args, device)

    episode_rewards = [0.0] * len(agent_order)  # 累积每个 agent 的 episode 奖励
    while env.agents:
        actions = {}
        for agent_name in env.agents:
            idx = agent_order.index(agent_name)
            obs = torch.as_tensor(observations[agent_name], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act = actors[idx](obs).squeeze(0)
            actions[agent_name] = act.cpu().numpy().clip(0.0, 1.0)

        observations, rewards, terminations, truncations, infos = env.step(actions)
        # print("rewards:", rewards)
        for i, agent_name in enumerate(env.agents):
            episode_rewards[i] += rewards[agent_name]

    mean_reward = sum(episode_rewards) / len(episode_rewards)

    env.close()

    print(f"Episode rewards: {episode_rewards}")