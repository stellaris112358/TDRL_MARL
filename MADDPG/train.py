"""
train.py — 基于 Hydra 配置的 MADDPG 训练入口

用法：
  # 默认配置（simple_spread_v3，训练模式）
  python train.py

  # 覆盖单个参数
  python train.py time_steps=500000 batch_size=128

  # 评估模式
  python train.py evaluate=true

  # 切换场景（conf/scenario/ 下需有对应 yaml）
  python train.py scenario=simple_spread_v3
"""

import os
import sys
from argparse import Namespace

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# 确保项目根目录在 sys.path 中（Hydra 可能改变 cwd）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from runner import Runner
from common.pettingzoo_wrapper import make_pz_env


def cfg_to_namespace(cfg: DictConfig) -> Namespace:
    """将 Hydra DictConfig 转换为 argparse.Namespace，供原有代码复用。
    将 scenario 等子分组的键展平到顶层，以兼容原有 args.xxx 调用方式。
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    flat: dict = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            # 展平子分组（如 scenario: {scenario_name: ...}）
            flat.update(v)
        else:
            flat[k] = v
    return Namespace(**flat)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("MADDPG 训练配置：")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    args = cfg_to_namespace(cfg)

    # 设置随机种子（可选，便于复现）
    seed = getattr(args, "seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env, args = make_pz_env(args)
    runner = Runner(args, env)

    if args.evaluate:
        returns = runner.evaluate()
        print(f"Average returns: {returns:.4f}")
    else:
        runner.run()


if __name__ == "__main__":
    main()
