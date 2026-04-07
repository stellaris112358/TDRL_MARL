# TDRL_MARL

This repository now keeps only the multi-agent TdRL code path:

- `MADDPG/train.py`: baseline MADDPG training
- `MADDPG/train_tdrl.py`: MADDPG + TdRL reward learning
- `reward_model/reward_model_ma_tdrl.py`: multi-agent return/reward model
- `tester/spread.py`: `simple_spread_v3` behavioral tests

The previous single-agent SAC/PPO, DMControl, MetaWorld, and SB3-based code has been removed to keep the project focused and easier to maintain.

## Quick Start

Recommended: Python 3.10+ and a recent PyTorch install that matches your CUDA environment.

```bash
pip install -r requirements.txt
cd MADDPG
python train_tdrl.py
```

Useful overrides:

```bash
python train_tdrl.py time_steps=1000000 tdrl.reward_update=100
python train_tdrl.py evaluate=true
python train.py
```

## Project Layout

```text
TDRL_MARL/
├── MADDPG/                  # Multi-agent training entrypoints and MADDPG implementation
├── reward_model/           # MATdRLRewardModel
├── tester/                 # Tester base class + simple_spread_v3 tests
├── fastkan/                # Optional FastKAN backbone for the return network
├── doc/                    # MARL-focused documentation
└── requirements.txt
```

## Docs

- `doc/TDRL_MADDPG_TRAINING.md`: training loop and reward-model update flow
- `MADDPG/README.md`: MADDPG module notes
