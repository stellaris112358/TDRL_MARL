"""
Multi-Agent TdRL Reward Model for MADDPG.

Key design:
  - Each agent has its own Reward Network: obs_i + act_i -> r_i
  - All agents share a single Return Network: joint_ind_test_vector -> predicted_return
  - Trajectory data is stored in joint format:
      inputs:  (traj_len, n_agents * (obs_dim + act_dim))
      s_nexts: (traj_len, n_agents * obs_dim)
  - The Tester operates on joint inputs/s_nexts.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
from scipy.stats import skew

try:
    from gymnasium.wrappers.utils import RunningMeanStd
except ImportError:
    # Minimal fallback to keep the MARL reward model self-contained.
    class RunningMeanStd:
        def __init__(self, shape=()):
            self.mean = np.zeros(shape, np.float64)
            self.var = np.ones(shape, np.float64)
            self.count = 1e-4
        def update(self, x):
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count
            self.mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
            self.var = M2 / tot_count
            self.count = tot_count

try:
    from fastkan import FastKAN
except ImportError:
    FastKAN = None

from tester import Tester


# ── gen_net (self-contained, avoid importing reward_model.py) ──
def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh', lipschitz=False):
    net = []
    for i in range(n_layers):
        if lipschitz:
            net.append(spectral_norm(nn.Linear(in_size, H)))
        else:
            net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    elif activation == 'softmax':
        net.append(nn.Softmax(dim=-1))
    elif activation is None:
        pass
    else:
        net.append(nn.ReLU())
    return net


class MATdRLRewardModel:
    """Multi-Agent TdRL Reward Model."""

    def __init__(self,
                 n_agents: int,
                 obs_dims: list,
                 act_dims: list,
                 device: str,
                 tester: Tester,
                 use_kan=False,
                 return_ensemble_size: int = 3,
                 ensemble_size: int = 3,
                 spectral_norm=False,
                 loss_combine="early_stop",
                 change_penalty=0.0,
                 mse_coef=1.0,
                 es_coef=10.0,
                 lr: float = 3e-4,
                 mb_size: int = 128,
                 size_segment: int = 25,
                 max_size: int = 200,
                 activation=None,
                 ):
        self.n_agents = n_agents
        self.obs_dims = obs_dims        # list: per-agent obs dim
        self.act_dims = act_dims        # list: per-agent act dim
        self.device = device
        self.tester = tester

        self.use_kan = use_kan
        self.return_ensemble_size = return_ensemble_size
        self.ensemble_size = ensemble_size
        self.spectral_norm = spectral_norm
        self.loss_combine = loss_combine
        self.change_penalty = change_penalty
        self.mse_coef = mse_coef
        self.es_coef = es_coef
        self.lr = lr
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.size_segment = size_segment
        self.max_size = max_size
        self.activation = activation

        # joint dims for tester
        self.joint_obs_dim = sum(obs_dims)
        self.joint_act_dim = sum(act_dims)

        # ── per-agent reward networks ──
        # reward_ensembles[agent_id][member] = network
        self.reward_ensembles = []
        self.reward_paramlsts = []
        self.reward_opts = []

        for agent_id in range(n_agents):
            agent_ensemble = []
            agent_params = []
            in_size = obs_dims[agent_id] + act_dims[agent_id]
            for _ in range(ensemble_size):
                model = nn.Sequential(*gen_net(
                    in_size=in_size, out_size=1, H=256, n_layers=3,
                    activation=activation)).float().to(device)
                agent_ensemble.append(model)
                agent_params.extend(model.parameters())
            self.reward_ensembles.append(agent_ensemble)
            self.reward_paramlsts.append(agent_params)
            self.reward_opts.append(torch.optim.Adam(agent_params, lr=lr))

        # ── shared return network ──
        self.ind_num = len(tester._ind_tests)
        self.return_ensemble = []
        self.return_parmlst = []

        for _ in range(return_ensemble_size):
            if use_kan:
                model = FastKAN([self.ind_num, 64, 64, 1]).to(device)
            else:
                model = nn.Sequential(*gen_net(
                    in_size=self.ind_num, out_size=1, H=256, n_layers=3,
                    activation=activation, lipschitz=spectral_norm,
                )).float().to(device)
            self.return_ensemble.append(model)
            self.return_parmlst.extend(model.parameters())

        self.return_opt = torch.optim.Adam(self.return_parmlst, lr=lr)
        self.return_params = []
        for member in range(return_ensemble_size):
            self.return_params.extend(list(self.return_ensemble[member].parameters()))

        # ── trajectory data (joint format) ──
        self.inputs = []    # list of traj arrays, each (T, joint_obs+joint_act)
        self.targets = []   # list of traj arrays, each (T, n_agents) real rewards
        self.s_nexts = []   # list of traj arrays, each (T, joint_obs)

        # ── test statistics ──
        self.pass_count = np.zeros(len(tester._pf_tests))
        self.ind_rms = RunningMeanStd(shape=self.ind_num)
        self.ind_skew = np.zeros(self.ind_num)
        self.ind_order = np.arange(self.ind_num)

    # ── data management ──────────────────────────────────────

    def add_data(self, obs_list, act_list, next_obs_list, rew_list, done):
        """
        Add a single timestep of multi-agent data.
        Args:
            obs_list:      list of np.array, one per agent
            act_list:      list of np.array, one per agent
            next_obs_list: list of np.array, one per agent
            rew_list:      list of float, one per agent
            done:          bool
        """
        joint_sa = np.concatenate([
            np.concatenate([obs_list[i], act_list[i]]) for i in range(self.n_agents)
        ]).reshape(1, -1)
        joint_s_next = np.concatenate(next_obs_list).reshape(1, -1)
        joint_rew = np.array(rew_list).reshape(1, -1)  # (1, n_agents)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(joint_sa)
            self.targets.append(joint_rew)
            self.s_nexts.append(joint_s_next)
        elif done:
            if isinstance(self.inputs[-1], list) and len(self.inputs[-1]) == 0:
                self.inputs[-1] = joint_sa
                self.targets[-1] = joint_rew
                self.s_nexts[-1] = joint_s_next
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], joint_sa])
                self.targets[-1] = np.concatenate([self.targets[-1], joint_rew])
                self.s_nexts[-1] = np.concatenate([self.s_nexts[-1], joint_s_next])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                self.s_nexts = self.s_nexts[1:]
            self.inputs.append([])
            self.targets.append([])
            self.s_nexts.append([])
        else:
            if isinstance(self.inputs[-1], list) and len(self.inputs[-1]) == 0:
                self.inputs[-1] = joint_sa
                self.targets[-1] = joint_rew
                self.s_nexts[-1] = joint_s_next
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], joint_sa])
                self.targets[-1] = np.concatenate([self.targets[-1], joint_rew])
                self.s_nexts[-1] = np.concatenate([self.s_nexts[-1], joint_s_next])

    # ── reward prediction ────────────────────────────────────

    def _extract_agent_sa(self, joint_sa, agent_id):
        """
        Extract agent_id's (obs, act) from joint_sa.
        joint_sa: (..., sum(obs_dims) + sum(act_dims))
        Returns: (..., obs_dims[agent_id] + act_dims[agent_id])
        """
        # Layout: [obs0, act0, obs1, act1, ..., obsN, actN]
        offset = 0
        for i in range(agent_id):
            offset += self.obs_dims[i] + self.act_dims[i]
        size = self.obs_dims[agent_id] + self.act_dims[agent_id]
        return joint_sa[..., offset:offset + size]

    def r_hat_member(self, agent_id, x, member=-1):
        """Predict reward for agent_id from its local (obs, act)."""
        return self.reward_ensembles[agent_id][member](
            torch.from_numpy(x).float().to(self.device))

    def r_hat(self, agent_id, joint_sa):
        """
        Mean reward prediction across ensemble for a single agent.
        joint_sa: (joint_obs+joint_act,) — single step
        Returns: float scalar
        """
        agent_sa = self._extract_agent_sa(np.asarray(joint_sa), agent_id)
        r_hats = []
        for member in range(self.ensemble_size):
            pred = self.r_hat_member(agent_id, agent_sa.reshape(1, -1), member)
            r_hats.append(pred.detach().cpu().numpy())
        return float(np.mean(r_hats))

    def r_hat_batch(self, agent_id, joint_sa_batch):
        """
        Mean reward prediction across ensemble for a batch.
        joint_sa_batch: (batch, joint_obs+joint_act)
        Returns: (batch, 1)
        """
        agent_sa = self._extract_agent_sa(joint_sa_batch, agent_id)
        r_hats = []
        for member in range(self.ensemble_size):
            pred = self.r_hat_member(agent_id, agent_sa, member).detach().cpu().numpy()
            r_hats.append(pred)
        return np.mean(np.array(r_hats), axis=0)

    # ── batch size scheduling ────────────────────────────────

    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size * new_frac)

    # ── preference generation ────────────────────────────────

    def get_preference(self, pf1, pf2, ind1, ind2):
        if np.sum(pf1) > np.sum(pf2):
            return 0
        elif np.sum(pf1) < np.sum(pf2):
            return 1
        elif np.sum(pf1) == len(pf1):
            return 0.5

        idx = np.argsort(self.pass_count)
        for i in idx:
            if pf1[i] > pf2[i]:
                return 0
            elif pf1[i] < pf2[i]:
                return 1

        if (ind1 > ind2).sum() == len(ind1):
            return 0
        elif (ind1 < ind2).sum() == len(ind1):
            return 1

        ind1_n = ind1 / np.sqrt(self.ind_rms.var + 1e-6)
        ind2_n = ind2 / np.sqrt(self.ind_rms.var + 1e-6)

        for i in self.ind_order:
            if ind1_n[i] != ind2_n[i]:
                return 0 if ind1_n[i] > ind2_n[i] else 1

        return 0.5

    def update_ind_skew(self, inds):
        for i in range(self.ind_num):
            self.ind_skew[i] = skew(inds[:, i])

    # ── training ─────────────────────────────────────────────

    def prepare_training(self):
        """Segment trajectories and compute test results."""
        input_segments = []
        for traj in self.inputs:
            if isinstance(traj, list) and len(traj) == 0:
                continue
            for start in range(0, len(traj), self.size_segment):
                input_segments.append(traj[start:start + self.size_segment])

        s_next_segments = []
        for s_next in self.s_nexts:
            if isinstance(s_next, list) and len(s_next) == 0:
                continue
            for start in range(0, len(s_next), self.size_segment):
                s_next_segments.append(s_next[start:start + self.size_segment])

        self.input_segments = np.array(input_segments)
        self.s_next_segments = np.array(s_next_segments)

        self.inds = self.tester.ind_test(self.input_segments, self.s_next_segments)
        self.ind_rms = RunningMeanStd(shape=self.ind_num)
        self.ind_rms.update(self.inds)
        self.update_ind_skew(self.inds)
        self.ind_order = np.argsort(self.ind_skew)[::-1]

        self.inds_torch = torch.from_numpy(self.inds).float().to(self.device)

        self.pfs = self.tester.pf_test(input_segments, s_next_segments)
        self.pass_count = np.sum(self.pfs, axis=0)

        ind_mean = np.mean(self.inds, axis=0)
        pf_mean = np.sum(self.pfs, axis=0) / self.pfs.shape[0]
        return ind_mean, pf_mean

    def train_returns(self):
        """Train the shared Return Network via preference-based CE loss."""
        max_len = len(self.input_segments)
        mb_size = min(self.mb_size, max_len)
        num_epochs = int(np.ceil(max_len / mb_size))
        ensemble_losses = [[] for _ in range(self.return_ensemble_size)]
        ensemble_acc = np.array([0.0 for _ in range(self.return_ensemble_size)])

        with torch.no_grad():
            ensemble_returns = []
            for member in range(self.return_ensemble_size):
                returns = self.return_ensemble[member](self.inds_torch)
                ensemble_returns.append(returns.detach().cpu().numpy())
            ensemble_returns = np.array(ensemble_returns)
            ensemble_returns = torch.from_numpy(ensemble_returns).float().to(self.device)

        total = 0
        for epoch in range(num_epochs):
            batch_index1 = np.random.choice(max_len, size=mb_size, replace=True)
            batch_index2 = np.random.choice(max_len, size=mb_size, replace=True)

            pf1 = self.pfs[batch_index1]
            pf2 = self.pfs[batch_index2]
            ind1 = self.inds[batch_index1]
            ind2 = self.inds[batch_index2]

            labels = np.array([
                self.get_preference(pf1[i], pf2[i], ind1[i], ind2[i])
                for i in range(mb_size)
            ])
            labels = torch.from_numpy(labels).float().to(self.device).reshape(-1, 1)

            inds1 = self.inds_torch[batch_index1]
            inds2 = self.inds_torch[batch_index2]
            ens_ret1 = ensemble_returns[:, batch_index1]
            ens_ret2 = ensemble_returns[:, batch_index2]

            ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            mse_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for member in range(self.return_ensemble_size):
                if member == 0:
                    total += torch.sum(labels != 0.5).cpu().item()

                ret_hat1 = self.return_ensemble[member](inds1)
                ret_hat2 = self.return_ensemble[member](inds2)
                ret_hat = torch.cat((ret_hat1, ret_hat2), dim=-1)

                log_probs = torch.log_softmax(ret_hat, dim=-1)
                label_probs = torch.cat([1 - labels, labels], dim=-1)
                curr_loss = -(log_probs * label_probs).sum(dim=-1).mean()

                change_loss = ((ens_ret1[member] - ret_hat1) ** 2
                               + (ens_ret2[member] - ret_hat2) ** 2).mean()

                ce_loss = ce_loss + curr_loss
                mse_loss = mse_loss + change_loss
                ensemble_losses[member].append(curr_loss.item())

                _, predicted = torch.max(ret_hat.data, -1)
                predicted = predicted.unsqueeze(-1)[labels != 0.5]
                real_labels = labels[labels != 0.5]
                correct = (predicted == torch.round(real_labels)).sum().item()
                ensemble_acc[member] += correct

            self.return_opt.zero_grad()

            if self.loss_combine == "weight":
                loss = ce_loss + self.change_penalty * mse_loss
                loss.backward()
                self.return_opt.step()
            elif self.loss_combine == "early_stop":
                grad_ce = torch.autograd.grad(ce_loss, self.return_params, retain_graph=True)
                grad_mse = torch.autograd.grad(mse_loss, self.return_params, retain_graph=True)
                norm_ce = torch.norm(torch.nn.utils.parameters_to_vector(grad_ce), p=2).clamp(min=1e-8)
                norm_mse = torch.norm(torch.nn.utils.parameters_to_vector(grad_mse), p=2).clamp(min=1e-8)
                if norm_mse > self.es_coef * norm_ce:
                    break
                total_grad = [g_ce + self.mse_coef * g_mse for g_ce, g_mse in zip(grad_ce, grad_mse)]
                for p, g in zip(self.return_params, total_grad):
                    p.grad = g
                self.return_opt.step()
            else:
                loss = ce_loss + self.change_penalty * mse_loss
                loss.backward()
                self.return_opt.step()

        if total == 0:
            total = 1
        ensemble_acc = ensemble_acc / total
        ensemble_mean_losses = np.array([
            np.mean(ensemble_losses[m]) if ensemble_losses[m] else 0.0
            for m in range(self.return_ensemble_size)
        ])
        return ensemble_acc, ensemble_mean_losses

    def train_reward(self):
        """
        Train per-agent Reward Networks.
        Each agent's reward network is supervised by the Return Network output,
        distributed equally across agents.
        """
        max_len = len(self.input_segments)
        mb_size = min(self.mb_size, max_len)
        num_epochs = int(np.ceil(max_len / mb_size))

        all_losses = {i: [] for i in range(self.n_agents)}

        for epoch in range(num_epochs):
            last_index = min((epoch + 1) * mb_size, max_len)
            idxs = np.random.choice(max_len, size=mb_size, replace=True)

            sa_t = self.input_segments[idxs]   # (mb, seg_len, joint_sa_dim)
            inds = self.inds_torch[idxs]

            # target return from return network
            with torch.no_grad():
                r_hats = []
                for m in range(self.return_ensemble_size):
                    r_hats.append(self.return_ensemble[m](inds))
                r_t = torch.stack(r_hats).mean(dim=0)  # (mb, 1)
                # distribute equally to each agent
                r_t_per_agent = r_t / self.n_agents

            for agent_id in range(self.n_agents):
                self.reward_opts[agent_id].zero_grad()
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                agent_sa = self._extract_agent_sa(sa_t, agent_id)  # (mb, seg_len, agent_sa_dim)

                for member in range(self.ensemble_size):
                    r_hat = self.reward_ensembles[agent_id][member](
                        torch.from_numpy(agent_sa).float().to(self.device))
                    r_hat_sum = r_hat.sum(axis=1)  # (mb, 1) sum over segment
                    curr_loss = nn.MSELoss()(r_hat_sum, r_t_per_agent)
                    loss = loss + curr_loss
                    all_losses[agent_id].append(curr_loss.item())

                loss.backward()
                self.reward_opts[agent_id].step()

        mean_losses = {i: np.mean(all_losses[i]) if all_losses[i] else 0.0
                       for i in range(self.n_agents)}
        return mean_losses

    # ── save / load ──────────────────────────────────────────

    def save(self, model_dir, step):
        import os
        for agent_id in range(self.n_agents):
            for member in range(self.ensemble_size):
                path = os.path.join(model_dir, f'reward_agent{agent_id}_m{member}_{step}.pt')
                torch.save(self.reward_ensembles[agent_id][member].state_dict(), path)
        for member in range(self.return_ensemble_size):
            path = os.path.join(model_dir, f'return_m{member}_{step}.pt')
            torch.save(self.return_ensemble[member].state_dict(), path)

    def load(self, model_dir, step):
        import os
        for agent_id in range(self.n_agents):
            for member in range(self.ensemble_size):
                path = os.path.join(model_dir, f'reward_agent{agent_id}_m{member}_{step}.pt')
                self.reward_ensembles[agent_id][member].load_state_dict(torch.load(path))
        for member in range(self.return_ensemble_size):
            path = os.path.join(model_dir, f'return_m{member}_{step}.pt')
            self.return_ensemble[member].load_state_dict(torch.load(path))
