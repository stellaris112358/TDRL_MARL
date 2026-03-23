"""PettingZoo Parallel API -> legacy MADDPG environment adapter."""

from __future__ import annotations

from importlib import import_module

import numpy as np


def _scenario_candidates(name: str):
    if not name.endswith("_v3"):
        yield f"{name}_v3"
    yield name


def _import_pz_module(scenario_name: str):
    last_error = None
    for candidate in _scenario_candidates(scenario_name):
        for prefix in ("pettingzoo.mpe", "mpe2"):
            module_name = f"{prefix}.{candidate}"
            try:
                module = import_module(module_name)
                if hasattr(module, "parallel_env"):
                    return module, candidate
            except ModuleNotFoundError as exc:
                last_error = exc
    if last_error is not None:
        raise last_error
    raise ModuleNotFoundError(f"Unable to resolve scenario '{scenario_name}'")


class PettingZooWrapper:
    """Wrap a PettingZoo ParallelEnv with the list-based interface used here."""

    def __init__(self, parallel_env):
        self._env = parallel_env
        self.metadata = getattr(parallel_env, "metadata", {})

        self._env.reset()
        self._refresh_agent_layout()
        self.n = len(self.agents)
        self.observation_space = [self._env.observation_space(agent) for agent in self.agents]
        self.action_space = [self._env.action_space(agent) for agent in self.agents]

    def _ordered_agents(self):
        base_agents = list(getattr(self._env, "agents", []))
        custom_order = list(getattr(self._env, "agent_order", []))
        possible_agents = list(getattr(self._env, "possible_agents", base_agents))
        if custom_order:
            return list(custom_order)
        if base_agents:
            return base_agents
        return possible_agents

    def _refresh_agent_layout(self):
        self.agents = self._ordered_agents()
        self.learning_agents = list(getattr(self._env, "learning_agents", self.agents))
        self.fixed_policy_agents = list(getattr(self._env, "fixed_policy_agents", []))

    def reset(self):
        obs_dict, _ = self._env.reset()
        self._refresh_agent_layout()
        return [obs_dict[agent] for agent in self.agents]

    def step(self, actions):
        if len(actions) != len(self.agents):
            raise ValueError(
                f"Expected {len(self.agents)} actions for agents {self.agents}, got {len(actions)}"
            )

        mapped_actions = [
            ((np.clip(action, -1.0, 1.0) + 1.0) / 2.0).astype(np.float32) for action in actions
        ]
        action_dict = {agent: mapped_actions[i] for i, agent in enumerate(self.agents)}

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(action_dict)
        self._refresh_agent_layout()

        obs_list = [
            obs_dict.get(agent, np.zeros(self.observation_space[i].shape, dtype=np.float32))
            for i, agent in enumerate(self.agents)
        ]
        reward_list = [float(rew_dict.get(agent, 0.0)) for agent in self.agents]
        done = any(term_dict.values()) or any(trunc_dict.values())
        return obs_list, reward_list, done, info_dict

    def render(self, mode=None, **kwargs):
        try:
            return self._env.render()
        except Exception:
            return None

    def close(self):
        if hasattr(self._env, "close"):
            return self._env.close()
        return None


def make_pz_env(args):
    """Create a PettingZoo MPE env and write back MADDPG-compatible metadata."""

    scenario_name = args.scenario_name
    module, resolved_scenario = _import_pz_module(scenario_name)

    render_mode = "human" if args.evaluate else None
    raw_env = module.parallel_env(
        max_cycles=args.max_episode_len,
        continuous_actions=True,
        render_mode=render_mode,
    )

    if getattr(args, "coop_tag", False) and resolved_scenario.startswith("simple_tag"):
        try:
            from .simple_tag_coop import SimpleTagCoopParallelEnv
        except Exception:
            from common.simple_tag_coop import SimpleTagCoopParallelEnv

        raw_env = SimpleTagCoopParallelEnv(raw_env)

    env = PettingZooWrapper(raw_env)

    args.n_players = env.n
    args.n_agents = len(getattr(env, "learning_agents", [])) or (env.n - args.num_adversaries)
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]
    args.action_shape = [
        env.action_space[i].shape[0] if hasattr(env.action_space[i], "shape") else env.action_space[i].n
        for i in range(args.n_agents)
    ]
    args.high_action = 1
    args.low_action = -1
    args.resolved_scenario_name = resolved_scenario

    return env, args
