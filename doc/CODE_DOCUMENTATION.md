# TdRL 代码说明文档

> 对应论文：Yu 等, 2025, *Test-driven Reinforcement Learning*

---

## 1. 项目概述

本项目实现了 **TdRL（Test-driven Reinforcement Learning，测试驱动强化学习）**。其核心思想是将软件工程中的**自动化测试**理念引入强化学习的奖励设计：

- 传统基于人类偏好的奖励学习（如 RLHF）需要大量人工标注；
- TdRL 改为由**预先设计的测试用例**自动生成偏好标签，无需人工干预；
- 测试用例分两类：**Pass-Fail (PF) 测试**（布尔型，行为是否达标）和**指示性轨迹测试 (Indicative Trajectory Test)**（连续值，量化行为质量）；
- 系统先训练 **Return Network**（将测试指标映射为预测回报），再以其为监督信号训练 **Reward Network**（输出即时奖励），最终用学到的奖励驱动 SAC 或 PPO 智能体。

新增：项目已扩展 **MARL 版本 TdRL**，在 MADDPG 上引入多智能体的 Return/Reward 联合训练与回放重标注。

---

## 2. 环境依赖与安装

### 系统要求

推荐：Ubuntu 24.04，Python ≥ 3.10，4 核 CPU，16 GB RAM，NVIDIA RTX 3060 或更高。

### MuJoCo 安装

```shell
wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
mkdir -p ~/.mujoco
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
rm mujoco210-linux-x86_64.tar.gz
```

在 `~/.bashrc` 或 `~/.zshrc` 中添加：

```shell
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
```

### Python 依赖（`requirements.txt`）

| 包 | 说明 |
|---|---|
| `stable-baselines3[extra]` | PPO/SAC 基础框架（本项目有定制修改） |
| `hydra-core` | 配置管理系统 |
| `gym` | 环境接口 |
| `transformers` | 学习率调度工具 |
| `scikit-image` | 图像处理辅助 |
| `metaworld` | MetaWorld 机器人操控任务集 |

```shell
conda create -n tdrl python=3.10
pip install -r requirements.txt
cd dmc2gym && pip install .
```

---

## 3. 项目目录结构总览

```
TDRL_MARL/
├── train_tdrl.py          # 主训练入口：SAC + TdRL 奖励学习
├── train_ppo_tdrl.py      # PPO + TdRL 奖励学习
├── train_sac.py           # 纯 SAC 基线
├── train_ppo.py           # 纯 PPO 基线
├── train_return.py        # 用真实回报监督奖励模型的基线
│
├── tester/                # 测试器模块：定义 PF 测试与指示性测试
├── reward_model/          # 奖励模型模块：Return Network + Reward Network
├── agent/                 # SAC 智能体（Actor/Critic）
├── stable_baselines3/     # 定制 SB3：PPO_CUSTOM / PPO_REWARD
├── fastkan/               # FastKAN 网络（可选替代 MLP）
├── conf/                  # Hydra 配置文件
├── utils/                 # 工具函数：环境创建、回放缓冲区、日志
└── dmc2gym/               # DMControl → Gymnasium 封装器

├── MADDPG/                # 多智能体训练（MADDPG + TdRL）
│   ├── train_tdrl.py       # MARL 版 TdRL 训练入口
│   ├── conf/               # Hydra 配置（含 TdRL 子配置）
│   ├── runner.py           # 训练/评估运行器（TdRL 扩展）
│   └── maddpg/             # MADDPG 核心算法
```

---

## 4. 核心模块详解

### 4.1 测试器模块 (`tester/`)

#### `Tester` 基类 (`tester.py`)

定义了所有测试器的接口规范：

- `_pf_tests`：Pass-Fail 测试列表，每项测试接收轨迹 `(inputs, s_nexts)` 返回 `bool`；
- `_ind_tests`：指示性轨迹测试列表，返回连续数值；
- `pf_test()` / `ind_test()`：批量执行所有测试；
- `@batch` 装饰器：自动支持批量轨迹输入，并内置 LRU 缓存以避免重复计算。

#### 环境特定测试器

| 类名 | 环境 | PF 测试 | 指示性测试 |
|---|---|---|---|
| `CartPole_Balance` | cartpole_balance | 杆直立、小车在中心区域 | 直立帧数、居中帧数 |
| `Walker_Stand` | walker_stand | 身高 > 1.2m、躯干直立 | 身高、直立度、速度惩罚 |
| `Walker_Walk` | walker_walk | 上述 + 均速 ≈ 1.5 m/s | 上述 + 均速 |
| `Walker_Run` | walker_run | 上述，目标速度 = 8 m/s | 同上 |
| `Walker_Jump` | walker_jump | 躯干直立、最大身高 > 5m | 直立、最大高度、速度惩罚 |
| `Cheetah_Run` | cheetah_run | 均速 ≥ 10 m/s | 均速 |
| `Quadruped_Walk/Run` | quadruped_* | 躯干直立、达到目标速度 | 直立度、均速 |

`TestDict`（`tester/__init__.py`）维护环境名到测试器类的映射，训练时通过 `TestDict[cfg.env]()` 实例化。

---

### 4.2 奖励模型模块 (`reward_model/`)

#### `RewardModel` 基类 (`reward_model.py`)

- **集成网络**：构建 `ensemble_size` 个独立 MLP 奖励网络（`gen_net`），通过 Adam 优化；
- **轨迹缓冲区**：以轨迹为单位存储 `(s,a)` 序列，FIFO 淘汰旧轨迹（最大容量 `max_size`）；
- **推理接口**：`r_hat(x)` 返回所有成员的平均即时奖励预测，`r_hat_batch()` 支持批量输入；
- **辅助工具**：`KCenterGreedy()` 用于多样性样本选择；`relabel_with_predictor()` 在回放缓冲区层面由奖励预测值替换原始奖励。

#### `TdRLRewardModel` (`reward_model_tdrl.py`) — 核心创新

继承自 `TestRewardModel`（`RewardModel` + `Tester` 的组合），实现论文核心算法：

**网络结构**

- **Reward Network**：`ensemble_size` 个 MLP，输入 `(s, a)`，输出即时奖励；
- **Return Network**：`return_ensemble_size` 个网络，输入为指示性测试向量（维度 = `len(_ind_tests)`），输出预测轨迹回报；可选用 MLP（默认）或 FastKAN（`use_kan=True`）。

**偏好生成 `get_preference(pf1, pf2, ind1, ind2)`**

按以下优先级比较两条轨迹的优劣：
1. 通过 PF 测试数量多的更优；
2. 若数量相同，先通过**历史最难**（通过率最低）的 PF 测试的更优；
3. 若 PF 完全相同，按指示性测试归一化分值逐项比较（按偏度排序，最能区分轨迹的指标优先）。

**训练流程**

```
prepare_training()        # 分段轨迹 → 计算所有轨迹的 ind 值和 pf 值
    ↓
train_returns()           # Return Network：CE 损失 + 变化惩罚 MSE 损失
    ↓
train_reward()            # Reward Network：以 Return Network 输出为监督目标的 MSE 损失
```

**损失组合策略（`loss_combine`）**

| 策略 | 描述 |
|---|---|
| `weight` | `loss = CE + λ·MSE`，λ 由 `change_penalty` 指定 |
| `grad_norm` | 归一化两项损失的梯度后加权合并 |
| `weighted_grad_norm` | 按梯度范数自适应加权 |
| `early_stop` | 当 MSE 梯度范数超过 CE 的 `es_coef` 倍时提前停止 |

**`train_reward_direct()`**：消融模式，跳过 Return Network，直接对 Reward Network 用 PF/ind 偏好标签做 CE 训练。

#### `ReturnRewardModel` (`reward_model_return.py`)

基线模型：直接用环境真实回报（MSE）监督奖励网络，不使用测试器。

#### `MATdRLRewardModel` (`reward_model_ma_tdrl.py`) — 多智能体扩展

用于 MADDPG 的多智能体 TdRL：

- **Reward Network（每个智能体独立）**：输入为该智能体本地 `(o_i, a_i)`；
- **Return Network（所有智能体共享）**：输入为**联合指示性测试向量**，输出预测轨迹回报；
- **联合轨迹存储**：轨迹以 joint 形式存储：
  - `inputs`: $(T, \sum_i (o_i + a_i))$
  - `s_nexts`: $(T, \sum_i o_i)$
  - `targets`: $(T, N)$（每个智能体的真实回报）
- **测试器**：`Tester` 接收 joint 轨迹，输出 PF/ind 测试指标；
- **训练流程**：`prepare_training()` → `train_returns()` → `train_reward()`，与单智能体 TdRL 对齐。

---

### 4.3 智能体模块 (`agent/`)

- **`Agent`**：抽象基类，定义 `reset()` / `train()` / `update()` / `act()` 接口；
- **`DiagGaussianActor`**：对角高斯策略，输出经 Tanh 压缩的 SquashedNormal 分布；
- **`DoubleQCritic`**：双 Q 网络，缓解过估计问题；
- **`SACAgent`**：完整 SAC 实现，包含：
  - 自动温度调节（`learnable_temperature`）；
  - 状态熵探索 `update_state_ent()`：无监督阶段使用 k-NN 距离估算状态熵作为内在奖励；
  - `reset_critic()`：在无监督探索结束后重置 Critic，避免错误 Q 值污染后续训练。

---

### 4.4 Stable Baselines3 定制模块 (`stable_baselines3/`)

本项目对 SB3 进行了扩展以支持在线奖励学习：

- **`PPO_CUSTOM`** (`ppo_custom.py`)：标准 PPO，用于对比实验基线；
- **`PPO_REWARD`** (`ppo_with_reward.py`)：在 PPO 训练循环中集成了 `TdRLRewardModel`，每隔 `num_interact` 步触发一次奖励模型更新；
- **`OnPolicyRewardAlgorithm`** (`on_policy_with_reward_algorithm.py`)：PPO_REWARD 的父类，负责：
  - `collect_rollouts()`：环境交互时同步收集轨迹、更新奖励模型、以预测奖励替换环境奖励写入 RolloutBuffer；
  - `collect_rollouts_unsuper()`：无监督探索阶段使用状态熵作为内在奖励驱动探索；
  - `learn_reward()`：触发 Return Network 与 Reward Network 的完整训练周期。

---

### 4.5 FastKAN 模块 (`fastkan/`)

实现了基于**径向基函数 (RBF)** 的 Kolmogorov-Arnold Network（KAN）的高效版本：

- `RadialBasisFunction`：用均匀网格上的高斯核作为样条基；
- `FastKANLayer`：RBF 样条线性变换 + 可选的基础线性变换（`use_base_update`）+ LayerNorm；
- `FastKAN`：多层堆叠，作为 Return Network 的可选替代（通过 `use_kan=True` 启用）。

---

## 5. 训练入口与流程

### 5.1 `train_tdrl.py` — SAC + TdRL（主实验）

**训练阶段划分：**

```
[0, num_seed_steps)             随机动作收集种子数据
[num_seed_steps,
 num_seed_steps+num_unsup_steps) SAC 使用状态熵内在奖励进行无监督探索
到达 num_unsup_steps 时          首次训练奖励模型，重置 Critic，更新 SAC
[num_unsup_steps, ...)           每隔 num_interact 步重新训练奖励模型并 relabel 缓冲区
```

**`learn_reward()` 内部流程：**
1. 调用 `reward_model.prepare_training()` 分段并计算测试值；
2. 循环 `return_update` 轮训练 Return Network；
3. 循环 `reward_update` 轮训练 Reward Network；
4. 调用 `replay_buffer.relabel_with_predictor()` 用新奖励函数重标记所有经验。

### 5.2 `train_ppo_tdrl.py` — PPO + TdRL

使用 SB3 的并行向量化环境（`n_envs` 个并行环境），奖励模型学习集成在 `PPO_REWARD` 内部，对外接口与标准 `model.learn()` 一致。

### 5.3 `train_sac.py` — 纯 SAC 基线

使用环境真实奖励，无奖励模型，结构与 `train_tdrl.py` 相同但去除奖励学习部分。

### 5.4 `train_ppo.py` — 纯 PPO 基线

使用 `PPO_CUSTOM`，与标准 SB3 PPO 行为一致。

### 5.5 `train_return.py` — 真实回报监督基线

使用 `ReturnRewardModel`，用环境真实回报做 MSE 监督学习奖励网络，再用该奖励驱动 SAC，验证"有了完美的奖励信号"时的性能上界。

### 5.6 `MADDPG/train_tdrl.py` — MADDPG + TdRL（MARL）

在 MADDPG 训练循环中集成 `MATdRLRewardModel`，核心流程：

1. 每步将多智能体数据喂入 `MATdRLRewardModel.add_data()`；
2. 每隔若干 episode 训练一次 Reward/Return Network；
3. `relabel_buffer=true` 时，用预测奖励替换 buffer 中的环境奖励。

**运行示例：**

```shell
cd MADDPG
python train_tdrl.py
python train_tdrl.py time_steps=1000000 tdrl.reward_update=100
python train_tdrl.py evaluate=true
```

---

## 6. 配置系统 (`conf/`)

本项目使用 **Hydra** 管理配置，支持 YAML 继承与命令行覆盖。

**配置继承关系：**

```
sac.yaml  ←  reward_model.yaml  ←  tdrl.yaml
ppo.yaml  ←  reward_model.yaml  ←  ppo_tdrl.yaml
```

**MADDPG 的 TdRL 配置：**

`MADDPG/conf/config_tdrl.yaml` 中的 `tdrl:` 节点提供多智能体奖励模型参数与训练频率，例如：

- `reward_update`：每 N 个 episode 训练一次 reward model
- `return_train_epochs` / `reward_train_epochs`
- `num_interact`：奖励模型启动前的交互步数
- `relabel_buffer`：是否用预测奖励重标注

**关键超参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `return_ensemble_size` | 3 | Return Network 集成数量 |
| `ensemble_size` | 3 | Reward Network 集成数量 |
| `size_segment` | 50 | 轨迹分段长度（步数） |
| `traj_max_size` | 100 | 轨迹缓冲区最大条数 |
| `num_interact` | 5000 | 奖励模型更新间隔步数 |
| `return_update` | 50 | 每次更新 Return Network 的轮数 |
| `reward_update` | 50 | 每次更新 Reward Network 的轮数 |
| `loss_combine` | `early_stop` | 损失组合策略 |
| `es_coef` | 10 | early_stop 策略的梯度范数阈值倍数 |
| `use_kan` | `false` | 是否使用 FastKAN 作为 Return Network |
| `rew_decomp` | `true` | 是否启用两阶段（Return→Reward）训练，`false` 时直接训练 |
| `acc_threshold` | 0.99 | Return Network 准确率早停阈值 |

**命令行覆盖示例：**

```shell
python train_tdrl.py env=walker_run seed=0 device=cuda num_train_steps=1000000
python train_tdrl.py env=cheetah_run loss_combine=weight change_penalty=0.1
```

结果自动保存至：`./exp_records/{env}/{algorithm}/{overrides}/{seed}/{timestamp}/`

---

## 7. 工具模块 (`utils/`)

- **`make_env()`**：将 `walker_run` 格式的字符串解析为 DMControl 域名和任务名，通过 `dmc2gym` 创建 Gymnasium 兼容环境；
- **`make_metaworld_env()`**：创建 MetaWorld 机器人操控环境（通过 `metaworld_` 前缀触发）；
- **`ReplayBuffer`**：固定容量的环形缓冲区，支持单步 `add()` 和批量 `add_batch()`，以及 `relabel_with_predictor()` 就地重标记奖励；
- **`Logger` / `VideoRecorder`**：封装 TensorBoard SummaryWriter，支持标量、直方图日志和 MP4 视频录制。

---

## 8. 算法总体流程

```
初始化：环境 + SACAgent + ReplayBuffer + TdRLRewardModel(tester)
          |
          ▼
[种子阶段] 随机动作填充缓冲区
          |
          ▼
[无监督探索] 用状态熵内在奖励驱动 SAC 广泛探索状态空间
          |
          ▼
[首次奖励学习]
  ┌─ prepare_training(): 轨迹分段 → 运行 pf_test() & ind_test() → 归一化统计
  ├─ train_returns():    成对轨迹 → get_preference() 生成标签 → 训练 Return Network
  └─ train_reward():     Return Network 输出作为目标 → 训练 Reward Network
          |
          ▼
relabel_with_predictor(): 用新 Reward Network 重标记 ReplayBuffer 中所有奖励
          |
          ▼
[正式训练] SAC 用预测奖励更新，每隔 num_interact 步重复奖励学习
          |
          ▼
周期评估：运行 pf_test() & ind_test() 记录测试通过率，保存最优模型
```

---

## 9. 支持的环境与任务

| 环境键名 | 说明 |
|---|---|
| `cartpole_balance` | DMControl 倒立摆平衡 |
| `walker_stand` | Walker 站立 |
| `walker_walk` | Walker 步行（目标 1.5 m/s） |
| `walker_run` | Walker 奔跑（目标 8 m/s） |
| `walker_jump` | Walker 跳跃（目标高度 5m） |
| `walker_jump_run` | Walker 跳跑复合任务 |
| `cheetah_run` | HalfCheetah 奔跑（目标 10 m/s） |
| `quadruped_walk` | 四足机器人步行 |
| `quadruped_run` | 四足机器人奔跑 |
| `metaworld_*` | MetaWorld 系列机器人操控任务（前缀 `metaworld_`） |

---

## 10. 扩展指南

### 添加新环境的测试器

1. 在 `tester/` 下新建文件，继承 `Tester` 基类；
2. 设置 `self.ds`（状态维度）和 `self.da`（动作维度）；
3. 使用 `@batch` 装饰器实现若干 `pf_*` 方法（返回 `bool`）和 `ind_*` 方法（返回 `float`）；
4. 将方法分别添加到 `self._pf_tests` 和 `self._ind_tests` 列表；
5. 在 `tester/__init__.py` 的 `TestDict` 中注册环境键名。

```python
# 示例
from tester import Tester, batch

class MyEnv_Task(Tester):
    def __init__(self):
        super().__init__()
        self.ds = 10   # 状态维度
        self.da = 3    # 动作维度
        self._pf_tests = [self.pf_my_criterion]
        self._ind_tests = [self.ind_my_metric]

    @batch
    def pf_my_criterion(self, inputs, s_nexts):
        return bool(s_nexts[:, 0].mean() > 0.5)

    @batch
    def ind_my_metric(self, inputs, s_nexts):
        return float(s_nexts[:, 0].mean())
```

### 自定义损失组合策略

在 `reward_model_tdrl.py` 的 `train_returns()` 方法中，按照现有 `if/elif` 分支添加新策略并在配置中用 `loss_combine=my_strategy` 启用。
