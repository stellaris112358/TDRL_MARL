from .reward_model import TestRewardModel, gen_net
from fastkan import FastKAN
from tester import Tester

import torch
from torch import nn
from gymnasium.wrappers.utils import RunningMeanStd
import numpy as np
from scipy.stats import skew


class TdRLRewardModel(TestRewardModel):
    # reward model of adptive weight rewards
    
    def __init__(self, 
                 ds:int,
                 da:int,
                 device:str,
                 tester:Tester,
                 use_kan=False,
                 return_ensemble_size:int=1,
                 spectral_norm=False,
                 loss_combine="weight",
                 change_penalty=0.0,
                 mse_coef=1.0,
                 es_coef=1.0,
                 **kwargs
                 ):
        
        self.use_kan = use_kan
        self.return_ensemble_size = return_ensemble_size
        self.spectral_norm = spectral_norm
        self.loss_combine = loss_combine
        self.change_penalty = change_penalty
        self.mse_coef = mse_coef
        self.es_coef = es_coef
        
        self.return_ensemble = []
        self.return_parmlst = []
        
        super().__init__(ds, da, device, tester, **kwargs)
        
        self.pass_count = np.zeros(len(self.tester._pf_tests))
        self.ind_num = len(self.tester._ind_tests)
        self.ind_rms = RunningMeanStd(shape=self.ind_num)
        self.ind_skew = np.zeros(self.ind_num)
        self.ind_order = np.arange(self.ind_num)
    
    def construct_ensemble(self):
        """
        initialize reward networks and return networks
        """
        for i in range(self.ensemble_size):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
        
        for i in range(self.return_ensemble_size):
            if self.use_kan:
                model = FastKAN([len(self.tester._ind_tests), 64, 64, 1]).to(self.device)
            else:
                model = nn.Sequential(*gen_net(in_size=len(self.tester._ind_tests),
                                           out_size=1, H=256, n_layers=3,
                                           activation=self.activation, lipschitz=self.spectral_norm,
                                           )).float().to(self.device)
            self.return_ensemble.append(model)
            self.return_parmlst.extend(model.parameters())

        self.return_opt = torch.optim.Adam(self.return_parmlst, lr = self.lr)
        
        self.return_params = []
        for member in range(self.return_ensemble_size):
            self.return_params.extend(list(self.return_ensemble[member].parameters()))
        

    
    def return_hat_member(self, inputs, s_nexts, member=0):
        inds = self.tester.ind_test(inputs, s_nexts)
        return self.return_ensemble[member](torch.from_numpy(inds).float().to(self.device))
    
    def return_hat(self, inputs, s_nexts):
        r_hats = []
        for member in range(self.return_ensemble_size):
            r_hats.append(self.return_hat_member(inputs, s_nexts, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def get_preference(self, pf1, pf2, ind1, ind2):
        # pass more pf-test is better
        if np.sum(pf1) > np.sum(pf2):
            return 0
        elif np.sum(pf1) < np.sum(pf2):
            return 1
        elif np.sum(pf1) == len(pf1):
            # all pass return 0.5
            return 0.5
        
        # pass more difficult pf-test if better
        idx = np.argsort(self.pass_count)
        ind_idx = []
        for i in idx:
            if pf1[i] > pf2[i]:
                return 0
            elif pf1[i] < pf2[i]:
                return 1
            elif pf1[i] == 0 and pf2[i] == 0:
                ind_idx.append(i)
        
        if (ind1>ind2).sum() == len(ind1):
            return 0
        elif (ind1<ind2).sum() == len(ind1):
            return 1
        
        ind1 = ind1 / np.sqrt(self.ind_rms.var + 1e-6)
        ind2 = ind2 / np.sqrt(self.ind_rms.var + 1e-6)
                    
        for i in self.ind_order:
            t1 = ind1[i]
            t2 = ind2[i]
            if t1 != t2:
                if t1 > t2:
                    return 0
                if t1 < t2:
                    return 1
    
        Warning("no preference")
        return 0.5
    
    def update_ind_skew(self, inds):
        for i in range(self.ind_num):
            self.ind_skew[i] = skew(inds[:, i])
    
    def prepare_training(self):
        input_segments = []
        for traj in self.inputs:
            for start in range(0, len(traj), self.size_segment):
                input_segments.append(traj[start:start+self.size_segment])
        s_next_segments = []
        for s_next in self.s_nexts:
            for start in range(0, len(s_next), self.size_segment):
                s_next_segments.append(s_next[start:start+self.size_segment])
        
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
        max_len = len(self.input_segments)
        mb_size = self.mb_size
        num_epochs = int(np.ceil(max_len/self.mb_size))
        ensemble_losses = [[] for _ in range(self.return_ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.return_ensemble_size)])
        
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

            labels = [self.get_preference(pf1[i], pf2[i], ind1[i], ind2[i]) 
                      for i in range(mb_size)]
            labels = np.array(labels)
            labels = torch.from_numpy(labels).float().to(self.device).reshape(-1,1)
            
            inds1 = self.inds_torch[batch_index1]
            inds2 = self.inds_torch[batch_index2]
            
            ensemble_returns1 = ensemble_returns[:,batch_index1]
            ensemble_returns2 = ensemble_returns[:,batch_index2]
            
            ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            mse_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        
            for member in range(self.return_ensemble_size):
                if member == 0:
                    total += torch.sum(labels!=0.5).cpu().item()
                
                return_hat1 = self.return_ensemble[member](inds1)
                return_hat2 = self.return_ensemble[member](inds2)

                return_hat = torch.cat((return_hat1, return_hat2), dim=-1)
                log_probs = torch.log_softmax(return_hat, dim=-1)
                label_probs = torch.cat([1-labels, labels], dim=-1)
                curr_loss = -(log_probs * label_probs).sum(dim=-1).mean()
                
                
                change_loss = (ensemble_returns1[member] - return_hat1) ** 2 \
                + (ensemble_returns2[member] - return_hat2) ** 2
                change_loss = change_loss.mean()
                
                ce_loss = ce_loss + curr_loss  
                mse_loss = mse_loss + change_loss
                
                ensemble_losses[member].append(curr_loss.item())

                _, predicted = torch.max(return_hat.data, -1)
                predicted = predicted.unsqueeze(-1)[labels!=0.5]
                real_labels = labels[labels!=0.5]
                correct = (predicted == torch.round(real_labels)).sum().item()
                ensemble_acc[member] += correct
            
            self.return_opt.zero_grad()
            
            if self.loss_combine == "weight":
                loss = ce_loss + self.change_penalty * mse_loss
                loss.backward()
                self.return_opt.step()
            elif self.loss_combine == "grad_norm":
                
                grad_ce = torch.autograd.grad(ce_loss, self.return_params, retain_graph=True)
                grad_mse = torch.autograd.grad(mse_loss, self.return_params, retain_graph=True)
                
                flat_grad_ce = torch.nn.utils.parameters_to_vector(grad_ce)
                flat_grad_mse = torch.nn.utils.parameters_to_vector(grad_mse)
                norm_ce = torch.norm(flat_grad_ce, p=2)
                norm_mse = torch.norm(flat_grad_mse, p=2)
                
                norm_ce = torch.clamp(norm_ce, min=1e-8)
                norm_mse = torch.clamp(norm_mse, min=1e-8)
                
                if norm_mse > norm_ce:
                    grad_mse_normalized = [g / norm_mse * norm_ce for g in grad_mse]
                    grad_ce_normalized = [g / norm_mse for g in grad_mse]
                else:
                    grad_mse_normalized = grad_mse
                grad_ce_normalized = grad_ce
                
                total_grad = [
                    g_ce + self.mse_coef * g_mse
                    for g_ce, g_mse in 
                    zip(grad_ce_normalized, grad_mse_normalized)
                ]
                
                for p, g in zip(self.return_params, total_grad):
                    p.grad = g
                
                self.return_opt.step()
            
            elif self.loss_combine == "weighted_grad_norm":
                grad_ce = torch.autograd.grad(ce_loss, self.return_params, retain_graph=True) # type: ignore
                grad_mse = torch.autograd.grad(mse_loss, self.return_params, retain_graph=True) # type: ignore
                
                flat_grad_ce = torch.nn.utils.parameters_to_vector(grad_ce)
                flat_grad_mse = torch.nn.utils.parameters_to_vector(grad_mse)
                norm_ce = torch.norm(flat_grad_ce, p=2)
                norm_mse = torch.norm(flat_grad_mse, p=2)
                
                norm_ce = torch.clamp(norm_ce, min=1e-8)
                norm_mse = torch.clamp(norm_mse, min=1e-8)
                
                mean_norm = (norm_ce + norm_mse) / 2
                
                weight_ce = norm_ce / mean_norm
                weight_mse = norm_mse / mean_norm
                
                self.return_opt.zero_grad()
                loss = weight_ce * ce_loss + weight_mse * mse_loss
                loss.backward()
                self.return_opt.step()
                
            elif self.loss_combine == "early_stop":
                grad_ce = torch.autograd.grad(ce_loss, self.return_params, retain_graph=True)
                grad_mse = torch.autograd.grad(mse_loss, self.return_params, retain_graph=True)
                
                flat_grad_ce = torch.nn.utils.parameters_to_vector(grad_ce)
                flat_grad_mse = torch.nn.utils.parameters_to_vector(grad_mse)
                norm_ce = torch.norm(flat_grad_ce, p=2)
                norm_mse = torch.norm(flat_grad_mse, p=2)
                
                norm_ce = torch.clamp(norm_ce, min=1e-8)
                norm_mse = torch.clamp(norm_mse, min=1e-8)
                
                if norm_mse > self.es_coef * norm_ce:
                    break
                
                total_grad = [
                    g_ce + self.mse_coef * g_mse
                    for g_ce, g_mse in 
                    zip(grad_ce, grad_mse)
                ]
                
                for p, g in zip(self.return_params, total_grad):
                    p.grad = g
                
                self.return_opt.step()
                
            else:
                raise ValueError(f"loss_combine must be weight or grad_norm but get {self.loss_combine}")
            
        ensemble_acc = ensemble_acc / total
        ensemble_mean_losses = np.array([np.mean(ensemble_losses[member]) for member in range(self.return_ensemble_size)])
        
        return ensemble_acc, ensemble_mean_losses
    
    def train_reward(self):
        
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = len(self.input_segments)
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.mb_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            last_index = (epoch+1)*self.mb_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.ensemble_size):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.mb_size:last_index]
                sa_t = self.input_segments[idxs]
                
                inds = self.inds_torch[idxs]
                
                with torch.no_grad():
                    r_hats = []
                    for m in range(self.return_ensemble_size):
                        r_hats.append(self.return_ensemble[m](inds))
                    r_hats = torch.stack(r_hats)
                    r_t = r_hats.mean(dim=0)
                
                if member == 0:
                    total += len(sa_t)
                
                # get logits
                r_hat = self.r_hat_member(sa_t, member=member)
                r_hat = r_hat.sum(axis=1)

                # compute loss
                curr_loss = nn.MSELoss()(r_hat, r_t)
                loss = loss + curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = 0
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        ensemble_mean_losses = np.array([np.mean(ensemble_losses[member]) for member in range(self.ensemble_size)])
        
        return ensemble_acc, ensemble_mean_losses
    
    def train_reward_direct(self):
        
        assert self.activation == "tanh", "Only tanh activation is supported"
        
        max_len = len(self.input_segments)
        mb_size = self.mb_size
        num_epochs = int(np.ceil(max_len/self.mb_size))
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        with torch.no_grad():
            ensemble_returns = []
            for member in range(self.ensemble_size):
                r_hat = self.r_hat_member(self.input_segments, member=member)
                returns = r_hat.sum(axis=1)
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

            labels = [self.get_preference(pf1[i], pf2[i], ind1[i], ind2[i]) 
                      for i in range(mb_size)]
            labels = np.array(labels)
            labels = torch.from_numpy(labels).float().to(self.device).reshape(-1,1)
            
            sa_t1 = self.input_segments[batch_index1]
            sa_t2 = self.input_segments[batch_index2]
            
            ensemble_returns1 = ensemble_returns[:,batch_index1]
            ensemble_returns2 = ensemble_returns[:,batch_index2]
            
            ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            mse_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        
            for member in range(self.return_ensemble_size):
                if member == 0:
                    total += torch.sum(labels!=0.5).cpu().item()
                
                r_hat1 = self.r_hat_member(sa_t1, member=member)
                return_hat1 = r_hat1.sum(axis=1)
                r_hat2 = self.r_hat_member(sa_t2, member=member)
                return_hat2 = r_hat2.sum(axis=1)

                return_hat = torch.cat((return_hat1, return_hat2), dim=-1)
                log_probs = torch.log_softmax(return_hat, dim=-1)
                label_probs = torch.cat([1-labels, labels], dim=-1)
                curr_loss = -(log_probs * label_probs).sum(dim=-1).mean()
                
                change_loss = (ensemble_returns1[member] - return_hat1) ** 2 \
                + (ensemble_returns2[member] - return_hat2) ** 2
                change_loss = change_loss.mean()
                
                ce_loss = ce_loss + curr_loss  
                mse_loss = mse_loss + change_loss
                
                ensemble_losses[member].append(curr_loss.item())

                _, predicted = torch.max(return_hat.data, -1)
                predicted = predicted.unsqueeze(-1)[labels!=0.5]
                real_labels = labels[labels!=0.5]
                correct = (predicted == torch.round(real_labels)).sum().item()
                ensemble_acc[member] += correct
            
            self.opt.zero_grad()
            loss = ce_loss
            loss.backward()
            self.opt.step()
           
        ensemble_acc = ensemble_acc / total
        ensemble_mean_losses = np.array([np.mean(ensemble_losses[member]) 
                                         for member in range(self.ensemble_size)])
        
        return ensemble_acc, ensemble_mean_losses
    