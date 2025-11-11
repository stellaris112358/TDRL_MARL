import os
import time
from collections import deque
import hydra
from omegaconf import DictConfig, OmegaConf

os.environ["MUJOCO_GL"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import utils
from pathlib import Path
from agent.sac import SACAgent
from utils.logger import Logger, VideoRecorder
from utils.replay_buffer import ReplayBuffer
from reward_model import TdRLRewardModel
from tester import TestDict


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
        self.video_recorder = VideoRecorder(Path.cwd() if cfg.save_video else None)
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg.env, cfg.seed)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg.env, cfg.seed)
            self.log_success = False
        
        
        obs_dim = self.env.observation_space.shape[0] # type: ignore
        action_dim = self.env.action_space.shape[0] # type: ignore
        action_range = [
            float(self.env.action_space.low.min()), # type: ignore
            float(self.env.action_space.high.max()) # type: ignore
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
        self.step = 0
        self.best_score = -np.inf
        
        tester = TestDict[cfg.env]()
        
        # instantiating the reward model
        self.reward_model = TdRLRewardModel(
            obs_dim,
            action_dim,
            self.cfg.device,
            tester,
            use_kan=cfg.use_kan,
            return_ensemble_size=cfg.return_ensemble_size,
            spectral_norm=cfg.spectral_norm,
            loss_combine=cfg.loss_combine,
            change_penalty=cfg.change_penalty,
            mse_coef=cfg.mse_coef,
            es_coef=cfg.es_coef,
            
            max_size=cfg.traj_max_size,
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
        
        inputs = []
        s_nexts = []
        for episode in range(num_eval_episodes):
            if episode == 0:
                is_record_video = True
            else:
                is_record_video = False

            episode_success = 0

            obs,_ = self.env.reset()
            inputs.append([])
            s_nexts.append([])
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            
            if is_record_video:
                self.video_recorder.init(self.env)

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                if obs is not None and action is not None:
                    inputs[-1].append(np.concatenate([np.asarray(obs), np.asarray(action)], axis=-1).reshape(-1))
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, extra = step_result
                else:
                    obs, reward, done, extra = step_result
                    terminated = done
                    truncated = False
                
                s_nexts[-1].append(obs.reshape(-1))
                done = terminated or truncated
                episode_reward += float(reward)
                true_episode_reward += float(reward)
                if self.log_success:
                    if (isinstance(extra, dict) and 'success' in extra):
                        episode_success = max(episode_success, extra['success'])
                    else:
                        Warning("No success in extra")
                if is_record_video:
                    self.video_recorder.record(self.env)
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            if is_record_video:
                self.video_recorder.save(f'{self.step}.mp4')
        
        for i in range(len(inputs)):
            inputs[i] = np.stack(inputs[i])
            s_nexts[i] = np.stack(s_nexts[i])
                
        average_episode_reward /= num_eval_episodes
        average_true_episode_reward /= num_eval_episodes
        if self.log_success:
            success_rate /= num_eval_episodes
            success_rate *= 100.0
        
        pf_test = self.reward_model.tester.pf_test(inputs,s_nexts).mean(axis=0)
        
        for i in range(len(pf_test)):
            self.logger._try_sw_log('eval/pf_test_' + str(i), pf_test[i], self.step)
        
        ind_test = self.reward_model.tester.ind_test(inputs,s_nexts).mean(axis=0)
        for i in range(len(ind_test)):
            self.logger._try_sw_log('eval/ind_test_' + str(i), ind_test[i], self.step)
        
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
        
        # save last model
        self.agent.save("last_model", "last")
        
        # save best model
        if average_episode_reward > self.best_score:
            self.best_score = average_episode_reward
            self.agent.save("best_model", "best")
        
    
    def learn_reward(self):
        total_acc = 0 
        
        # pre computing for computing efficiency
        ind_mean, pf_mean = self.reward_model.prepare_training()
        for i in range(len(ind_mean)):
            self.logger._try_sw_log('train/ind_' + str(i), ind_mean[i], self.step)
        for i in range(len(pf_mean)):
            self.logger._try_sw_log('train/pf_' + str(i), pf_mean[i], self.step)
        
        if self.cfg.rew_decomp:
            # record max ind skew index
            if len(self.reward_model.ind_order):
                self.logger._try_sw_log('train/ind_order_0', self.reward_model.ind_order[0], self.step)
            
            # update return
            for epoch in range(self.cfg.return_update):
                return_acc, return_mean_losses = self.reward_model.train_returns()
                total_acc = np.mean(return_acc)
                self.logger._try_sw_log('train/total_acc', total_acc, self.step+epoch)
                for i in range(self.cfg.return_ensemble_size):
                    self.logger._try_sw_log('train/return_loss_' + str(i), return_mean_losses[i], self.step+epoch)
                if total_acc > self.cfg.acc_threshold:
                    break
            
            for epoch in range(self.cfg.reward_update):
                train_acc, ensemble_mean_losses = self.reward_model.train_reward()
                for i in range(self.cfg.ensemble_size):
                    self.logger._try_sw_log('train/rew_loss_' + str(i), ensemble_mean_losses[i], self.step+epoch)
                    
            print("Reward function is updated!! ACC: " + str(total_acc))
        else:
            for epoch in range(self.cfg.return_update):
                return_acc, return_mean_losses = self.reward_model.train_reward_direct()
                total_acc = np.mean(return_acc)
                self.logger._try_sw_log('train/total_acc', total_acc, self.step+epoch)
                for i in range(self.cfg.return_ensemble_size):
                    self.logger._try_sw_log('train/return_loss_' + str(i), return_mean_losses[i], self.step+epoch)
                if total_acc > self.cfg.acc_threshold:
                    break
            print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 

        interact_count = 0
        obs = None  # Ensure obs is always defined
        episode_step = 0  # Initialize episode_step to avoid unbound errors
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
                self.logger.log('train/traj num', len(self.reward_model.inputs), self.step)
                
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
                
                # first learn reward
                self.learn_reward()
                
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
            
            # 3 differences from above: update method (reset critic)
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
                        
                    self.learn_reward()
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                    interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            step_result = self.env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, extra = step_result
            else:
                next_obs, reward, done, extra = step_result
                terminated = done
                truncated = False
            done = terminated or truncated
            
            if obs is not None and action is not None:
                reward_hat = self.reward_model.r_hat(np.concatenate([np.asarray(obs), np.asarray(action)], axis=-1))
            else:
                reward_hat = 0.0

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if truncated else done
            episode_reward += reward_hat
            true_episode_reward += float(reward)
            
            if self.log_success :
                if (isinstance(extra, dict) and 'success' in extra):
                    episode_success = max(episode_success, extra['success'])
                else:
                    Warning("No success in extra")
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, next_obs, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1


@hydra.main(config_path='conf', config_name='tdrl', version_base=None)
def main(cfg: DictConfig):
    start_time = time.time()
    print(OmegaConf.to_yaml(cfg))
    workspace = Workspace(cfg)
    workspace.run()
    
if __name__ == '__main__':
    main()
