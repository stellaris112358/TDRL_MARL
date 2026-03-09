# MADDPG — 多智能体深度确定性策略梯度算法

基于 PyTorch 实现的 **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** 算法，运行在 [Multi-Agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs) 上。

> 论文：[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) (Lowe et al., 2017)

---

## 目录

- [项目概述](#项目概述)
- [算法原理](#算法原理)
- [项目结构](#项目结构)
- [模块详解](#模块详解)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [训练流程](#训练流程)
- [评估模式](#评估模式)
- [注意事项](#注意事项)

---

## 项目概述

本项目实现了 MADDPG 算法，用于多智能体环境中的协作/竞争任务。核心思想是 **集中式训练、分布式执行（Centralized Training with Decentralized Execution, CTDE）**：

- **训练时**：每个智能体的 Critic 网络可以访问所有智能体的观测和动作（全局信息）。
- **执行时**：每个智能体的 Actor 网络仅根据自身的局部观测做出决策。

当前已在 `simple_tag`（捕食者-猎物追逐）场景上完成预训练，并提供了模型存档。

---

## 算法原理

MADDPG 是 DDPG 在多智能体场景下的扩展，每个智能体 $i$ 维护独立的 Actor-Critic 网络：

- **Actor** $\mu_{\theta_i}(o_i)$：根据智能体自身观测 $o_i$ 输出连续动作。
- **Critic** $Q_{\phi_i}(o_1, \ldots, o_N, a_1, \ldots, a_N)$：根据所有智能体的观测和动作输出联合 Q 值。

**Critic 损失**（最小化 TD 误差）：

$$L(\phi_i) = \mathbb{E}\left[\left(Q_{\phi_i}(\mathbf{o}, \mathbf{a}) - y\right)^2\right], \quad y = r_i + \gamma \, Q_{\phi_i'}(\mathbf{o'}, \mathbf{a'})$$

**Actor 损失**（最大化期望回报）：

$$\nabla_{\theta_i} J \approx -\mathbb{E}\left[\nabla_{\theta_i} \mu_{\theta_i}(o_i) \, \nabla_{a_i} Q_{\phi_i}(\mathbf{o}, \mathbf{a})\big|_{a_i=\mu_{\theta_i}(o_i)}\right]$$

**目标网络**通过软更新保持稳定：

$$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$$

---

## 项目结构

```
MADDPG/
├── main.py                  # 程序主入口
├── agent.py                 # 智能体封装（动作选择 + 学习）
├── runner.py                # 训练/评估运行器
├── README.md                # 项目说明文档
├── common/
│   ├── arguments.py         # 命令行参数定义
│   ├── replay_buffer.py     # 经验回放缓冲区
│   └── utils.py             # 工具函数（环境创建等）
├── maddpg/
│   ├── actor_critic.py      # Actor / Critic 神经网络定义
│   └── maddpg.py            # MADDPG 核心算法（训练 + 软更新 + 保存）
└── model/
    └── simple_tag/          # 预训练模型存档
        ├── agent_0/
        ├── agent_1/
        └── agent_2/
```

---

## 模块详解

### `main.py` — 程序入口

- 解析命令行参数，创建环境，实例化 `Runner`。
- 根据 `--evaluate` 标志切换 **训练模式** 或 **评估模式**。

### `agent.py` — 智能体

| 方法 | 说明 |
|------|------|
| `select_action(o, noise_rate, epsilon)` | **ε-贪心 + 高斯噪声** 动作选择策略。以概率 ε 随机采样，否则通过 Actor 网络生成动作并叠加高斯噪声以实现探索。 |
| `learn(transitions, other_agents)` | 将采样的 transitions 和其他智能体传给 MADDPG 进行训练。 |

### `runner.py` — 训练/评估运行器

| 方法 | 说明 |
|------|------|
| `run()` | 主训练循环。按 step 推进环境，收集经验存入 Buffer，达到 batch_size 后开始训练，定期评估并保存学习曲线图。噪声和 ε 随训练步数线性衰减。 |
| `evaluate()` | 评估模式：关闭噪声和 ε，运行若干 episode 并返回平均回报。 |

### `common/arguments.py` — 超参数

通过 `argparse` 定义所有训练超参数（详见下方[命令行参数](#命令行参数)）。

### `common/replay_buffer.py` — 经验回放

- 以字典形式存储每个智能体的 $(o, u, r, o')$ 转移元组。
- 使用线程锁保证并发安全。
- 缓冲区满后随机覆盖旧经验（循环存储）。

### `common/utils.py` — 工具函数

| 函数 | 说明 |
|------|------|
| `store_args` | 装饰器，将函数参数自动存为实例属性。 |
| `make_env(args)` | 根据场景名加载 MPE 环境，自动设置 `n_players`、`n_agents`、`obs_shape`、`action_shape` 等参数。 |

### `maddpg/actor_critic.py` — 网络结构

| 网络 | 输入 | 隐藏层 | 输出 |
|------|------|--------|------|
| **Actor** | 单个智能体的观测 $o_i$ | 3 层全连接 (64) + ReLU | `tanh` 输出连续动作，范围 $[-1, 1]$ |
| **Critic** | 所有智能体的观测拼接 + 所有动作拼接 | 3 层全连接 (64) + ReLU | 标量 Q 值 |

### `maddpg/maddpg.py` — 核心算法

| 功能 | 说明 |
|------|------|
| 网络初始化 | 创建 Actor/Critic 及其目标网络，使用 Adam 优化器。 |
| 模型加载 | 启动时自动检测并加载已有模型参数。 |
| `train()` | 从 Buffer 采样 → 计算目标 Q → 更新 Critic → 更新 Actor → 软更新目标网络 → 定期保存模型。 |
| `_soft_update_target_network()` | 以系数 τ 对目标网络进行软更新。 |
| `save_model()` | 按训练步数保存 Actor/Critic 参数。 |

---

## 环境要求

| 依赖 | 版本建议 |
|------|---------|
| Python | >= 3.6 |
| PyTorch | >= 1.1.0 |
| NumPy | >= 1.16 |
| Matplotlib | >= 3.0 |
| tqdm | 任意 |
| [MPE](https://github.com/openai/multiagent-particle-envs) | 最新 |

安装 MPE：

```bash
git clone https://github.com/openai/multiagent-particle-envs.git
cd multiagent-particle-envs
pip install -e .
```

---

## 快速开始

### 训练

```bash
python main.py --scenario-name=simple_tag --time-steps=2000000
```

训练过程中：
- 模型定期保存至 `./model/simple_tag/agent_*/`
- 学习曲线图保存至 `./model/simple_tag/plt.png`
- 评估回报数据保存至 `./model/simple_tag/returns.pkl`

### 评估（使用预训练模型）

```bash
python main.py --scenario-name=simple_tag --evaluate-episodes=10 --evaluate=True
```

将加载预训练模型运行 10 个 episode，输出平均回报。

---

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--scenario-name` | str | `simple_tag` | MPE 场景名称 |
| `--max-episode-len` | int | 100 | 单个 episode 的最大步数 |
| `--time-steps` | int | 2,000,000 | 总训练步数 |
| `--num-adversaries` | int | 1 | 对手数量（不受 MADDPG 控制的智能体） |
| `--lr-actor` | float | 1e-4 | Actor 学习率 |
| `--lr-critic` | float | 1e-3 | Critic 学习率 |
| `--epsilon` | float | 0.1 | ε-贪心的初始探索概率 |
| `--noise_rate` | float | 0.1 | 高斯噪声初始比率 |
| `--gamma` | float | 0.95 | 折扣因子 |
| `--tau` | float | 0.01 | 目标网络软更新系数 |
| `--buffer-size` | int | 500,000 | 回放缓冲区容量 |
| `--batch-size` | int | 256 | 训练批量大小 |
| `--save-dir` | str | `./model` | 模型保存目录 |
| `--save-rate` | int | 2,000 | 每隔多少训练步保存一次模型 |
| `--evaluate-episodes` | int | 10 | 评估 episode 数 |
| `--evaluate-episode-len` | int | 100 | 评估 episode 长度 |
| `--evaluate` | bool | False | 是否进入评估模式 |
| `--evaluate-rate` | int | 1,000 | 训练中每隔多少步评估一次 |

---

## 训练流程

```
1. 解析参数 → 创建 MPE 环境 → 初始化 Runner
2. 主循环（共 time_steps 步）:
   ├── 每 max_episode_len 步重置环境
   ├── 各智能体选择动作（ε-贪心 + 高斯噪声）
   ├── 对手采用随机策略
   ├── 环境执行一步，获得 (s', r, done)
   ├── 将转移存入 Replay Buffer
   ├── 当 Buffer 数据量 ≥ batch_size:
   │   └── 每个智能体采样并训练（Actor + Critic + 软更新）
   ├── 每 evaluate_rate 步进行一次评估并绘制曲线
   └── 线性衰减 noise_rate 和 epsilon
3. 保存回报数据
```

---

## 评估模式

评估模式下：
- 噪声和 ε 均设为 0 → 智能体完全利用学到的策略。
- 渲染环境画面（`env.render()`）。
- 输出每个 episode 的回报及平均回报。

---

## 注意事项

- 提供的预训练模型（`model/simple_tag/`）并非最优，仅用于演示。可继续训练以获得更好的性能。
- `simple_tag` 场景中共 4 个智能体：3 个捕食者（由 MADDPG 控制）+ 1 个猎物（随机策略）。
- MPE 默认使用稀疏奖励；如需密集奖励，将 `multiagent-particle-envs/multiagent/scenarios/simple_tag.py` 中的 `shape=False` 改为 `shape=True`。
- Critic 网络输入为所有智能体的观测和动作拼接，因此不同场景中智能体数量或观测维度变化时网络结构会自动适配。
- 噪声和 ε 在训练过程中从 0.1 线性衰减至 0.05，以平衡探索与利用。
