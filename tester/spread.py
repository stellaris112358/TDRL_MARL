"""
Tester for PettingZoo MPE simple_spread_v3 (N=3 agents, 3 landmarks).

Observation per agent (18 dims):
  [0:2]   self_vel          (2)
  [2:4]   self_pos          (2)
  [4:10]  landmark_rel_pos  (3 landmarks * 2 = 6)
  [10:14] other_agent_rel_pos (2 other agents * 2 = 4)
  [14:18] comms             (2 other agents * 2 = 4)

For multi-agent TdRL we concatenate all agents' (obs, act) into a single
joint input vector and all agents' next_obs into a single joint s_next
vector so that the Tester interface remains the same:
  inputs shape:  (traj_len, n_agents * (obs_dim + act_dim))  = (T, 3*(18+5)) = (T, 69)
  s_nexts shape: (traj_len, n_agents * obs_dim)              = (T, 3*18)     = (T, 54)
"""

from tester import Tester, batch
import numpy as np
from scipy.optimize import linear_sum_assignment


class Spread_v3(Tester):

    def __init__(self, n_agents=3, n_landmarks=3, obs_dim=18, act_dim=5):
        super().__init__()
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.obs_dim_per_agent = obs_dim
        self.act_dim_per_agent = act_dim

        self.ds = n_agents * obs_dim        # 54
        self.da = n_agents * act_dim        # 15

        # thresholds
        self._cover_dist = 0.15       # distance to count as "covering" a landmark
        self._collision_dist = 0.038  # agent radius * 2 ≈ 0.075, half for overlap
        self._cover_ratio = 0.8       # fraction of steps that need coverage

        self._pf_tests = [
            self.pf_all_covered,
            self.pf_no_collision,
        ]

        self._ind_tests = [
            self.ind_avg_min_dist,
            self.ind_coverage_count,
            self.ind_collision_count,
        ]

    # ── helpers ──────────────────────────────────────────────

    def _parse_s_nexts(self, s_nexts: np.ndarray):
        """
        Parse joint s_nexts into per-agent arrays.
        Returns:
          agent_positions: (traj_len, n_agents, 2)
          landmark_rel_positions: (traj_len, n_agents, n_landmarks, 2)
          landmark_abs_positions: (traj_len, n_landmarks, 2) — computed from agent 0
        """
        T = s_nexts.shape[0]
        # split into per-agent obs
        agent_obs = s_nexts.reshape(T, self.n_agents, self.obs_dim_per_agent)

        # agent absolute positions: obs[2:4]
        agent_pos = agent_obs[:, :, 2:4]  # (T, n_agents, 2)

        # landmark relative positions from each agent: obs[4:10] reshaped to (n_landmarks, 2)
        lm_rel = agent_obs[:, :, 4:4 + self.n_landmarks * 2].reshape(
            T, self.n_agents, self.n_landmarks, 2)  # (T, n_agents, n_landmarks, 2)

        # absolute landmark positions computed from agent 0
        # landmark_abs = agent0_pos + landmark_rel_from_agent0
        lm_abs = agent_pos[:, 0:1, :] + lm_rel[:, 0, :, :]  # (T, n_landmarks, 2)

        return agent_pos, lm_rel, lm_abs

    def _compute_agent_landmark_dists(self, agent_pos, lm_abs):
        """
        Returns: (T, n_agents, n_landmarks) distance matrix
        """
        # agent_pos: (T, n_agents, 2), lm_abs: (T, n_landmarks, 2)
        diff = agent_pos[:, :, np.newaxis, :] - lm_abs[:, np.newaxis, :, :]  # (T, A, L, 2)
        return np.linalg.norm(diff, axis=-1)  # (T, A, L)

    def _compute_agent_dists(self, agent_pos):
        """
        Returns: (T, n_pairs) pairwise distances between agents
        """
        dists = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                d = np.linalg.norm(agent_pos[:, i, :] - agent_pos[:, j, :], axis=-1)
                dists.append(d)
        return np.stack(dists, axis=-1)  # (T, n_pairs)

    def _optimal_assignment_coverage(self, dist_matrix):
        """
        For a single timestep dist_matrix (n_agents, n_landmarks),
        use Hungarian algorithm to find best assignment,
        return the number of agents within _cover_dist of their assigned landmark.
        """
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        assigned_dists = dist_matrix[row_ind, col_ind]
        return np.sum(assigned_dists < self._cover_dist)

    # ── PF tests ─────────────────────────────────────────────

    @batch
    def pf_all_covered(self, inputs: np.ndarray, s_nexts: np.ndarray):
        """
        Pass if, for most timesteps, every landmark is covered by
        at least one agent (via optimal assignment).
        """
        agent_pos, _, lm_abs = self._parse_s_nexts(s_nexts)
        dist_mat = self._compute_agent_landmark_dists(agent_pos, lm_abs)  # (T, A, L)
        T = dist_mat.shape[0]
        covered_steps = 0
        for t in range(T):
            n_covered = self._optimal_assignment_coverage(dist_mat[t])
            if n_covered >= self.n_landmarks:
                covered_steps += 1
        return covered_steps / T >= self._cover_ratio

    @batch
    def pf_no_collision(self, inputs: np.ndarray, s_nexts: np.ndarray):
        """
        Pass if no inter-agent collisions occur throughout the trajectory.
        """
        agent_pos, _, _ = self._parse_s_nexts(s_nexts)
        pair_dists = self._compute_agent_dists(agent_pos)  # (T, n_pairs)
        return bool(np.all(pair_dists > self._collision_dist))

    # ── Indicative tests ─────────────────────────────────────

    @batch
    def ind_avg_min_dist(self, inputs: np.ndarray, s_nexts: np.ndarray):
        """
        Average over timesteps of: for optimal assignment, mean assigned distance.
        Lower is better. We return negative so that higher = better.
        """
        agent_pos, _, lm_abs = self._parse_s_nexts(s_nexts)
        dist_mat = self._compute_agent_landmark_dists(agent_pos, lm_abs)
        T = dist_mat.shape[0]
        total = 0.0
        for t in range(T):
            row_ind, col_ind = linear_sum_assignment(dist_mat[t])
            total += dist_mat[t][row_ind, col_ind].mean()
        avg = total / T
        return -avg  # negative: closer is better

    @batch
    def ind_coverage_count(self, inputs: np.ndarray, s_nexts: np.ndarray):
        """
        Total number of timesteps * landmarks that are covered.
        """
        agent_pos, _, lm_abs = self._parse_s_nexts(s_nexts)
        dist_mat = self._compute_agent_landmark_dists(agent_pos, lm_abs)
        T = dist_mat.shape[0]
        total = 0
        for t in range(T):
            n_covered = self._optimal_assignment_coverage(dist_mat[t])
            total += n_covered
        return total

    @batch
    def ind_collision_count(self, inputs: np.ndarray, s_nexts: np.ndarray):
        """
        Negative count of collisions (higher = fewer collisions = better).
        """
        agent_pos, _, _ = self._parse_s_nexts(s_nexts)
        pair_dists = self._compute_agent_dists(agent_pos)
        collisions = np.sum(pair_dists < self._collision_dist)
        return -collisions
