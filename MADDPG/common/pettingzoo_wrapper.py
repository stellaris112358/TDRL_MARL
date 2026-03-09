"""
PettingZoo Parallel API → 旧版 MPE 接口适配器

将 PettingZoo parallel_env 封装成与原项目 runner.py / agent.py 兼容的接口：
  - env.reset()  -> list of observations
  - env.step(actions)  -> (obs_list, reward_list, done, info)
  - env.n  -> total agent count
  - env.observation_space[i]
  - env.action_space[i]

注意：PettingZoo simple_spread_v3(continuous_actions=True) 的动作范围为 [0, 1]，
     而原 MADDPG Actor 输出 tanh ∈ [-1, 1]。
     本 wrapper 在 step() 内将动作从 [-1, 1] 线性映射到 [0, 1]，
     算法侧无需任何改动。
"""

import numpy as np
from importlib import import_module


class PettingZooWrapper:
    """将 PettingZoo Parallel API 封装为旧版 MPE 风格接口。"""

    def __init__(self, parallel_env):
        self._env = parallel_env
        # 初始化一次以获取 agents 列表（PettingZoo 要求先 reset）
        _obs, _ = self._env.reset()
        self.agents = list(self._env.agents)
        self.n = len(self.agents)
        self.observation_space = [self._env.observation_space(a) for a in self.agents]
        self.action_space = [self._env.action_space(a) for a in self.agents]

    # ── 旧版 MPE 接口 ──────────────────────────────────────

    def reset(self):
        obs_dict, _ = self._env.reset()
        return [obs_dict[a] for a in self.agents]

    def step(self, actions):
        """
        actions: list of np.ndarray, 每个元素范围 [-1, 1]（MADDPG 侧约定）
        内部映射到 PettingZoo 要求的 [0, 1]
        """
        # [-1, 1] -> [0, 1]，强制转为 float32 避免 dtype 校验警告
        mapped = [((np.clip(a, -1.0, 1.0) + 1.0) / 2.0).astype(np.float32) for a in actions]
        action_dict = {agent: mapped[i] for i, agent in enumerate(self.agents)}

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(action_dict)

        # 若某 agent 已结束则用零向量填充观测
        obs_list = [
            obs_dict.get(a, np.zeros(self.observation_space[i].shape, dtype=np.float32))
            for i, a in enumerate(self.agents)
        ]
        r_list = [float(rew_dict.get(a, 0.0)) for a in self.agents]
        done = any(term_dict.values()) or any(trunc_dict.values())
        return obs_list, r_list, done, info_dict

    def render(self):
        try:
            self._env.render()
        except Exception:
            # 无显示器（headless）环境下忽略渲染异常
            pass


def make_pz_env(args):
    """
    根据 args.scenario_name 动态加载 PettingZoo MPE 场景，
    并将所有环境相关参数（obs_shape、action_shape 等）写回 args。
    """
    scenario = args.scenario_name  # e.g. "simple_spread_v3"

    # 动态导入，兼容 pettingzoo.mpe 和 mpe2 等不同命名
    try:
        mod = import_module(f"pettingzoo.mpe.{scenario}")
    except ModuleNotFoundError:
        mod = import_module(f"mpe2.{scenario}")

    render_mode = "human" if args.evaluate else None
    raw_env = mod.parallel_env(
        max_cycles=args.max_episode_len,
        continuous_actions=True,
        render_mode=render_mode,
    )
    env = PettingZooWrapper(raw_env)

    # 写入 args（与原 make_env 保持字段一致）
    args.n_players = env.n
    # simple_spread_v3 全合作；若有对手可在 config 中设置 num_adversaries
    args.n_agents = env.n - args.num_adversaries
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
    # continuous Box(5,) 用 shape[0]；Discrete 用 .n
    args.action_shape = [
        env.action_space[i].shape[0] if hasattr(env.action_space[i], 'shape')
        else env.action_space[i].n
        for i in range(args.n_agents)
    ]
    args.high_action = 1
    args.low_action = -1

    return env, args
