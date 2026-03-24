from .reward_model import RewardModel, gen_net

import torch
from torch import nn
from gymnasium import Env
import numpy as np


class ReturnRewardModel(RewardModel):
    # reward model of adptive weight rewards
    
    def __init__(self, 
                 ds:int,
                 da:int,
                 device:str="cpu",
                 **kwargs
                 ):
        super().__init__(ds, da, device, **kwargs)
    
    def construct_ensemble(self):
        """
        initialize reward networks
        """
        for i in range(self.ensemble_size):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
    
    def weight_member(self, x, member=-1):
        weights = self.ensemble[member](torch.from_numpy(x).float().to(self.device))
        return weights
    
    def weight(self, x):
        weights = []
        for member in range(self.ensemble_size):
            weights.append(self.weight_member(x, member=member).detach().cpu().numpy())
        weights = np.array(weights)
        return np.mean(weights, axis=0)
    
    def train_reward(self):
        input_segments = []
        for traj in self.inputs:
            for start in range(0, len(traj), self.size_segment):
                input_segments.append(traj[start:start+self.size_segment])
        target_segments = []
        for rew in self.targets:
            for start in range(0, len(rew), self.size_segment):
                target_segments.append(rew[start:start+self.size_segment])
        
        input_segments = np.array(input_segments)
        target_segments = np.array(target_segments)
        
        rewards = torch.from_numpy(target_segments).float().to(self.device)
        
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        
        max_len = len(input_segments)
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
                sa_t = input_segments[idxs]
                r_t = rewards[idxs].sum(dim=1)
                
                if member == 0:
                    total += len(sa_t)
                
                # get logits
                r_hat = self.r_hat_member(sa_t, member=member)
                r_hat = r_hat.sum(axis=1)

                # compute loss
                curr_loss = nn.MSELoss()(r_hat, r_t)
                loss += curr_loss
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