import gym
import os
import hydra
import time
from omegaconf import DictConfig, OmegaConf

os.environ["MUJOCO_GL"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import utils
from stable_baselines3 import PPO_CUSTOM
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_dmcontrol_env, make_vec_metaworld_env
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.vec_env import VecNormalize


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

@hydra.main(config_path='conf', config_name='ppo', version_base=None)
def main(cfg: DictConfig):
    start_time = time.time()
    print(OmegaConf.to_yaml(cfg))
    
    metaworld_flag = False
    if 'metaworld' in cfg.env:
        metaworld_flag = True
    
    clip_range = linear_schedule(cfg.clip_init)
    
    # Parallel environments
    if metaworld_flag:
        env = make_vec_metaworld_env(
            cfg.env, 
            n_envs=cfg.n_envs, 
            seed=cfg.seed)
    else:
        env = make_vec_dmcontrol_env(    
            cfg.env, 
            n_envs=cfg.n_envs, 
            seed=cfg.seed)
    
    if cfg.normalize:
        env = VecNormalize(env, norm_reward=False)
    
    # network arch
    net_arch = (dict(pi=[cfg.hidden_dim]*cfg.num_layer, 
                     vf=[cfg.hidden_dim]*cfg.num_layer))
    policy_kwargs = dict(net_arch=net_arch)
    
    work_dir = os.getcwd()
    # train model
    model = PPO_CUSTOM(
        MlpPolicy, env,
        tensorboard_log=work_dir,
        save_tb=cfg.save_tb, 
        agent_name=cfg.agent_name,
        seed=cfg.seed, 
        learning_rate=cfg.lr,
        batch_size=cfg.batch_size,
        n_steps=cfg.n_steps,
        ent_coef=cfg.ent_coef,
        policy_kwargs=policy_kwargs,
        use_sde=cfg.use_sde,
        sde_sample_freq=cfg.sde_freq,
        gae_lambda=cfg.gae_lambda,
        clip_range=clip_range,
        n_epochs=cfg.n_epochs,
        metaworld_flag=metaworld_flag,
        verbose=1,
        device=cfg.device,
        )
    
    model.learn(total_timesteps=cfg.num_train_steps)

if __name__ == "__main__":
    main()
    
    
    