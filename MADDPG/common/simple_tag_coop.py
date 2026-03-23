"""Cooperative wrapper for PettingZoo MPE ``simple_tag_v3``.

This wrapper keeps the original environment dynamics but changes the training
setup to fit the MADDPG code in this repository:

- only the chasers (adversaries) are treated as learning agents;
- the prey is controlled by a deterministic heuristic policy;
- all learning agents receive the same shared team reward.

The wrapper also exposes an explicit ``agent_order`` so downstream code can
consume observations and actions in a stable "learners first, fixed-policy
agents last" layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CoopTagConfig:
    prey_agent: str = "agent_0"
    team_reward: str = "sum_adversary"
    boundary_soft_limit: float = 0.80
    boundary_gain: float = 0.90
    nearest_chaser_gain: float = 1.25
    obstacle_gain: float = 0.70
    velocity_damping: float = 0.20
    obstacle_avoid_radius: float = 0.45
    multi_chaser_radius: float = 0.80


class SimpleTagCoopParallelEnv:
    """ParallelEnv wrapper that turns simple_tag into a chaser team task."""

    def __init__(self, base_parallel_env, cfg: Optional[CoopTagConfig] = None):
        self._env = base_parallel_env
        self.cfg = cfg or CoopTagConfig()
        self.metadata = getattr(base_parallel_env, "metadata", {})
        self.unwrapped = getattr(base_parallel_env, "unwrapped", base_parallel_env)
        self.prey_agent = self.cfg.prey_agent

        self._world = getattr(self.unwrapped, "world", None)
        self._landmark_count = self._infer_landmark_count()

        obs, _ = self._env.reset()
        self._last_obs = obs
        self._refresh_agent_views()

        self._obs_spaces = {
            agent: self._env.observation_space(agent) for agent in self.possible_agents
        }
        self._act_spaces = {
            agent: self._env.action_space(agent) for agent in self.possible_agents
        }

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _infer_landmark_count(self) -> int:
        if self._world is None:
            return 2
        return sum(1 for landmark in self._world.landmarks if not getattr(landmark, "boundary", False))

    def _refresh_agent_views(self) -> None:
        current_agents = list(getattr(self._env, "agents", []))
        possible_agents = list(getattr(self._env, "possible_agents", current_agents))

        if self.cfg.prey_agent not in possible_agents:
            raise ValueError(
                f"Configured prey agent '{self.cfg.prey_agent}' not found in environment agents: {possible_agents}"
            )

        if current_agents:
            stable_agents = list(current_agents)
        else:
            stable_agents = [agent for agent in possible_agents if agent != self.prey_agent]
            if self.prey_agent in possible_agents:
                stable_agents.append(self.prey_agent)

        self.team_agents = [agent for agent in stable_agents if agent != self.prey_agent]
        self.learning_agents = list(self.team_agents)
        self.fixed_policy_agents = [agent for agent in stable_agents if agent == self.prey_agent]

        ordered_possible = [agent for agent in possible_agents if agent != self.prey_agent]
        if self.prey_agent in possible_agents:
            ordered_possible.append(self.prey_agent)

        self.agent_order = list(self.learning_agents) + list(self.fixed_policy_agents)
        self.agents = list(self.agent_order)
        self.possible_agents = ordered_possible

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    def state(self):
        if hasattr(self._env, "state"):
            return self._env.state()
        raise AttributeError("Base env does not support state().")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._refresh_agent_views()
        return obs, info

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return np.zeros_like(vec)
        return vec / norm

    def _boundary_avoidance(self, position: np.ndarray) -> np.ndarray:
        force = np.zeros(2, dtype=np.float32)
        soft = self.cfg.boundary_soft_limit
        gain = self.cfg.boundary_gain

        if position[0] > soft:
            force[0] -= gain * (position[0] - soft) / max(1.0 - soft, 1e-6)
        elif position[0] < -soft:
            force[0] += gain * (-soft - position[0]) / max(1.0 - soft, 1e-6)

        if position[1] > soft:
            force[1] -= gain * (position[1] - soft) / max(1.0 - soft, 1e-6)
        elif position[1] < -soft:
            force[1] += gain * (-soft - position[1]) / max(1.0 - soft, 1e-6)

        return force

    def _parse_prey_observation(
        self, prey_obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        obs = np.asarray(prey_obs, dtype=np.float32).reshape(-1)
        if obs.size < 4:
            raise ValueError(f"Prey observation too short: shape={obs.shape}")

        velocity = obs[0:2]
        position = obs[2:4]
        cursor = 4

        obstacle_rel_positions: List[np.ndarray] = []
        for _ in range(self._landmark_count):
            if cursor + 2 > obs.size:
                break
            obstacle_rel_positions.append(obs[cursor : cursor + 2])
            cursor += 2

        remaining = obs.size - cursor
        other_agent_count = max(0, len(self.possible_agents) - 1)
        chaser_count = min(other_agent_count, remaining // 2)

        chaser_rel_positions: List[np.ndarray] = []
        for _ in range(chaser_count):
            if cursor + 2 > obs.size:
                break
            chaser_rel_positions.append(obs[cursor : cursor + 2])
            cursor += 2

        return velocity, position, obstacle_rel_positions, chaser_rel_positions

    def _fallback_prey_rule(self, prey_obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(prey_obs, dtype=np.float32).reshape(-1)
        rel = obs[-2:] if obs.size >= 2 else np.zeros(2, dtype=np.float32)
        direction = self._unit(-rel)
        return self._direction_to_action(direction)

    def _direction_to_action(self, direction: np.ndarray) -> np.ndarray:
        move = np.asarray(direction, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(move))
        if norm > 1.0:
            move = move / norm

        action = np.array(
            [
                0.0,
                max(0.0, -float(move[0])),
                max(0.0, float(move[0])),
                max(0.0, -float(move[1])),
                max(0.0, float(move[1])),
            ],
            dtype=np.float32,
        )
        return np.clip(action, 0.0, 1.0)

    def _prey_rule(self, prey_obs: np.ndarray) -> np.ndarray:
        try:
            velocity, position, obstacle_rel_positions, chaser_rel_positions = self._parse_prey_observation(
                prey_obs
            )
        except ValueError:
            return self._fallback_prey_rule(prey_obs)

        direction = np.zeros(2, dtype=np.float32)

        if chaser_rel_positions:
            dists = np.array([np.linalg.norm(rel) for rel in chaser_rel_positions], dtype=np.float32)
            nearest_index = int(np.argmin(dists))
            nearest_rel = chaser_rel_positions[nearest_index]
            direction += self.cfg.nearest_chaser_gain * self._unit(-nearest_rel)

            for rel, dist in zip(chaser_rel_positions, dists):
                if dist < self.cfg.multi_chaser_radius:
                    pressure = 1.0 - (dist / max(self.cfg.multi_chaser_radius, 1e-6))
                    direction += 0.35 * pressure * self._unit(-rel)

        for rel in obstacle_rel_positions:
            dist = float(np.linalg.norm(rel))
            if dist < self.cfg.obstacle_avoid_radius:
                pressure = 1.0 - (dist / max(self.cfg.obstacle_avoid_radius, 1e-6))
                direction += self.cfg.obstacle_gain * pressure * self._unit(-rel)

        direction += self._boundary_avoidance(position)
        direction -= self.cfg.velocity_damping * velocity

        if np.linalg.norm(direction) < 1e-6:
            direction = self._unit(-position)

        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)

        return self._direction_to_action(direction)

    def _make_team_reward(self, rewards: Dict[str, float]) -> float:
        adversary_rewards = [float(rewards.get(agent, 0.0)) for agent in self.team_agents]
        if not adversary_rewards:
            return 0.0
        if self.cfg.team_reward == "mean_adversary":
            return float(np.mean(adversary_rewards))
        return float(np.sum(adversary_rewards))

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        active_agents = list(getattr(self._env, "agents", []))
        action_dict = dict(actions)

        if self.prey_agent in active_agents:
            prey_obs = self._last_obs.get(self.prey_agent) if isinstance(self._last_obs, dict) else None
            if prey_obs is not None:
                action_dict[self.prey_agent] = self._prey_rule(prey_obs)

        filtered_actions = {agent: action_dict[agent] for agent in active_agents if agent in action_dict}
        current_learning_agents = [agent for agent in active_agents if agent != self.prey_agent]
        obs, rewards, terminations, truncations, infos = self._env.step(filtered_actions)

        self._last_obs = obs

        team_reward = self._make_team_reward(rewards)
        coop_rewards = dict(rewards)
        for agent in current_learning_agents:
            coop_rewards[agent] = team_reward

        self._refresh_agent_views()

        return obs, coop_rewards, terminations, truncations, infos
