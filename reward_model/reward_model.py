import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils.video import VideoRecorder
from gymnasium import Env
from typing import Optional

from tester import Tester


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

def KCenterGreedy(obs, full_obs, num_new_sample, device):
    selected_index = []
    current_index = np.arange(obs.shape[0])
    new_obs = obs
    new_full_obs = full_obs
    start_time = time.time()
    for count in range(num_new_sample):
        dist = compute_smallest_dist(new_obs, new_full_obs, device)
        max_index = torch.argmax(dist)
        max_index = max_index.item()
        
        if count == 0:
            selected_index.append(max_index)
        else:
            selected_index.append(current_index[max_index])
        current_index = current_index[0:max_index] + current_index[max_index+1:]
        
        new_obs = obs[current_index]
        new_full_obs = np.concatenate([
            full_obs, 
            obs[selected_index]], 
            axis=0)
    return selected_index

def compute_smallest_dist(obs, full_obs, device):
    obs = torch.from_numpy(obs).float()
    full_obs = torch.from_numpy(full_obs).float()
    batch_size = 100
    with torch.no_grad():
        total_dists = []
        for full_idx in range(len(obs) // batch_size + 1):
            full_start = full_idx * batch_size
            if full_start < len(obs):
                full_end = (full_idx + 1) * batch_size
                dists = []
                for idx in range(len(full_obs) // batch_size + 1):
                    start = idx * batch_size
                    if start < len(full_obs):
                        end = (idx + 1) * batch_size
                        dist = torch.norm(
                            obs[full_start:full_end, None, :].to(device) - full_obs[None, start:end, :].to(device), dim=-1, p=2
                        )
                        dists.append(dist)
                dists = torch.cat(dists, dim=1)
                small_dists = torch.min(dists, dim=1).values
                total_dists.append(small_dists)
                
        total_dists = torch.cat(total_dists)
    return total_dists.unsqueeze(1)


class RewardModel:
    def __init__(self, 
                 ds:int,
                 da:int,
                 device:str="cpu",
                 ensemble_size:int=3, 
                 lr:float=3e-4, 
                 mb_size:int=128, 
                 size_segment:int=1, 
                 max_size=100, 
                 activation:str='tanh', 
                 ):
        """

        Args:
            ds (int): dimension of state
            da (int): dimension of action
            ensemble_size (int, optional): number of reward networks. Defaults to 3.
            lr (float, optional): learning rate. Defaults to 3e-4.
            mb_size (int, optional): minibatch size. Defaults to 128.
            size_segment (int, optional): size of each segment. Defaults to 1.
            env_maker (_type_, optional): _description_. Defaults to None.
            max_size (int, optional): _description_. Defaults to 100.
            activation (str, optional): activation function of reward network. Defaults to 'tanh'.
        """
        
        # train data is trajectories, must process to sa and s.. 
        self.device = device  
        self.ds = ds
        self.da = da
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.ensemble = []  # list of reward networks
        self.paramlst = []  # list of reward network parameters
        self.opt = None # optimizer
        self.max_size = max_size    # max size of trajectory
        self.activation = activation
        self.size_segment = size_segment

        # initialize reward networks
        self.construct_ensemble()
        
        self.inputs = []    # list of trajectories, each item contains a list of (s,a) in a trajectory
        self.targets = []   # list of trajectory rewards, each item contains a list of rewards for (s,a) in a trajectory
        self.s_nexts = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.CEloss = nn.CrossEntropyLoss()

    
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        """
        change batch size to new_frac*self.origin_mb_size
        
        Args:
            new_frac (float): new batch size
        """
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch_size(self, new_batch):
        """
        set batch size to new_batch
        
        Args:
            new_batch (int): new batch size
        """
        self.mb_size = int(new_batch)
        
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

    def add_data(self, obs, act, next_obs, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        s_next = next_obs
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)
        flat_s_next = s_next.reshape(1, self.ds)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
            self.s_nexts.append(flat_s_next)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            self.s_nexts[-1] = np.concatenate([self.s_nexts[-1], flat_s_next])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                self.s_nexts = self.s_nexts[1:]
            self.inputs.append([])
            self.targets.append([])
            self.s_nexts.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
                self.s_nexts[-1] = flat_s_next
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                self.s_nexts[-1] = np.concatenate([self.s_nexts[-1], flat_s_next])
    
    def add_data_batch(self, obses, rewards, s_nexts):
        num_env = obses.shape[0]
        for index in range(num_env):
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                self.s_nexts = self.s_nexts[1:]
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
            self.s_nexts.append(s_nexts[index])
        
    def get_rank_probability(self, x_1, x_2):
        """
        get probability x_1 > x_2
        
        Args:
            x_1 (tensor): input 1
            x_2 (tensor): input 2
        
        Returns:
            (float, float): mean, std of the probabilities
        """
        # get probability x_1 > x_2
        probs = []
        for member in range(self.ensemble_size):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.ensemble_size):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], dim=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], dim=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(dim=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)

    def r_hat_batch_grad(self, x):
        # they say they average the rewards from each member of the ensemble,
        # but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member))
        r_hats = torch.stack(r_hats)  # (self.de, n_env, 1, 1)

        return torch.mean(r_hats, dim=0)  # (n_env, 1, 1)
    
    def save(self, model_dir, step):
        for member in range(self.ensemble_size):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.ensemble_size):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    
    def prepare_training(self):
        return None, None


class TestRewardModel(RewardModel):
    def __init__(self, 
                 ds, 
                 da, 
                 device,
                 tester:Tester, 
                 **kwargs
                 ):
        
        self.tester = tester
        super().__init__(ds, da, device, **kwargs)
    
    
        