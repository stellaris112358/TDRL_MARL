import argparse
import os
import time
from collections import deque
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import utils
from agent.sac import SACAgent
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer
from reward_model import ReturnRewardModel


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent_name,
            train_log_name=cfg.train_log_name,
            eval_log_name=cfg.eval_log_name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg.env, cfg.seed)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg.env, cfg.seed)
            self.log_success = False
        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = SACAgent(
            obs_dim, action_dim, action_range, cfg
        )

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        
        # instantiating the reward model
        self.reward_model = ReturnRewardModel(
            obs_dim,
            action_dim,
            self.cfg.device,
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.size_segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            )
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        num_eval_episodes = self.cfg.num_eval_episodes
        
        for episode in range(num_eval_episodes):
            obs,_ = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
                
        average_episode_reward /= num_eval_episodes
        average_true_episode_reward /= num_eval_episodes
        if self.log_success:
            success_rate /= num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        self.logger.log('eval/num_eval_episodes', num_eval_episodes,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
        self.total_feedback += self.reward_model.mb_size
        total_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                train_acc, ensemble_mean_losses = self.reward_model.train_reward()
                for i in range(self.cfg.ensemble_size):
                    self.logger._try_sw_log('train/ensemble_loss_' + str(i), ensemble_mean_losses[i], self.step+epoch)
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break
                    
        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            
            # if done, log & evaluate & reset
            if done:
                if self.step > 0:
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                # self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs,_ = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.size_segment / self.env._max_episode_steps)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            
            # 3 differences from above: first_flag, corner case, update method (reset critic)
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function

                if interact_count == self.cfg.num_interact:
                    # update schedule
                    if self.cfg.reward_schedule == 1:
                        frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                        if frac == 0:
                            frac = 0.01
                    elif self.cfg.reward_schedule == 2:
                        frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                    else:
                        frac = 1
                    self.reward_model.change_batch(frac)
                    
                    # update margin --> not necessary / will be updated soon
                    new_margin = np.mean(avg_train_true_return) * (self.cfg.size_segment / self.env._max_episode_steps)
                        
                    self.learn_reward()
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                    interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, terminated, truncated, extra = self.env.step(action)
            done = terminated or truncated
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, next_obs, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1


@hydra.main(config_path='conf', config_name='learn_return')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
