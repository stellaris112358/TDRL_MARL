# Return 网络与 Reward 网络详解

> 本文档覆盖单智能体版本（`TdRLRewardModel`）和多智能体版本（`MATdRLRewardModel`）中两类网络的设计目标、结构、训练方式及相互关系。

---

## 目录

- [Return 网络与 Reward 网络详解](#return-网络与-reward-网络详解)
  - [目录](#目录)
  - [1. 总体架构](#1-总体架构)
  - [2. Return 网络](#2-return-网络)
    - [2.1 作用](#21-作用)
    - [2.2 输入：指示性测试向量（ind tests）](#22-输入指示性测试向量ind-tests)
    - [2.3 网络结构](#23-网络结构)
    - [2.4 偏好标签生成](#24-偏好标签生成)
    - [2.5 训练目标（CE Loss）](#25-训练目标ce-loss)
    - [2.6 稳定性：Change Penalty（MSE Loss）](#26-稳定性change-penaltymse-loss)
    - [2.7 梯度合并策略（loss\_combine）](#27-梯度合并策略loss_combine)
  - [3. Reward 网络](#3-reward-网络)
    - [3.1 作用](#31-作用)
    - [3.2 输入：局部状态-动作对](#32-输入局部状态-动作对)
    - [3.3 网络结构](#33-网络结构)
    - [3.4 训练目标（MSE Loss）](#34-训练目标mse-loss)
    - [3.5 多智能体扩展：奖励分配](#35-多智能体扩展奖励分配)
  - [4. 两网络的协作关系](#4-两网络的协作关系)
  - [5. 数据流全景](#5-数据流全景)
  - [6. 超参数说明](#6-超参数说明)
  - [7. 单智能体 vs. 多智能体对比](#7-单智能体-vs-多智能体对比)

---

## 1. 总体架构

TdRL（Test-driven Reinforcement Learning）的核心思想是：**用行为测试（Behavioral Tests）代替人工奖励函数**，通过测试结果驱动 Reward 信号的生成。整个奖励建模分为两个串联的网络：

```
轨迹片段 (segment)
      │
      ▼
  Tester  ───────────► ind_test 指标向量 (K 维)
      │                              │
      │ pf_test 通过/失败向量         │
      │       │                      ▼
      │       └──────────► Return 网络 f_φ
      │                    (ind → 预测累积回报 R̂)
      │                              │
      │                              │ MSE 监督
      │                              ▼
      └──────────────────► Reward 网络 r_θ
                           (s, a → 单步奖励 r̂)
```

**Return 网络**负责将"测试分数"映射为轨迹的累积回报估计；  
**Reward 网络**负责将单步 `(s, a)` 映射为即时奖励，以 Return 网络的输出为监督信号。

---

## 2. Return 网络

### 2.1 作用

Return 网络 $f_\varphi$ 回答的问题是：

> **"这段轨迹的测试表现，对应多大的累积回报？"**

它是一个从**测试指标空间 → 标量回报**的映射，通过**轨迹片段间的偏好比较**来训练，不需要人工标注奖励。

---

### 2.2 输入：指示性测试向量（ind tests）

在调用 `prepare_training()` 时，轨迹库中每段轨迹 $\tau$ 都被送入 Tester，计算一个 $K$ 维的**指示性测试（Indicative Test）**向量：

$$\mathbf{v}(\tau) = \bigl[\text{ind}_1(\tau),\ \text{ind}_2(\tau),\ \ldots,\ \text{ind}_K(\tau)\bigr] \in \mathbb{R}^K$$

以 `simple_spread_v3` 为例，$K=3$，三个指标分别为：

| 索引 | 测试函数 | 含义 | 数值方向 |
|------|---------|------|----------|
| 0 | `ind_avg_min_dist` | 最优分配下智能体到地标的平均最近距离（取负） | 越大越好（越接近 0）|
| 1 | `ind_coverage_count` | 全轨迹中被覆盖的地标时间步数之和 | 越大越好 |
| 2 | `ind_collision_count` | 全轨迹碰撞次数（取负） | 越大越好（越接近 0）|

Return 网络的输入维度即为 $K$：

```python
# TdRLRewardModel
in_size = len(self.tester._ind_tests)   # K

# MATdRLRewardModel
self.ind_num = len(tester._ind_tests)   # K = 3
```

---

### 2.3 网络结构

**默认架构（MLP）**：

```
Linear(K → 256) → LeakyReLU
Linear(256 → 256) → LeakyReLU
Linear(256 → 256) → LeakyReLU
Linear(256 → 1)                ← 无激活函数（activation=None）
```

代码对应 `gen_net(in_size=K, out_size=1, H=256, n_layers=3, activation=None)`。

**可选架构（KAN）**：当 `use_kan=True` 时，使用 FastKAN 替换 MLP：

```
FastKAN([K, 64, 64, 1])
```

**集成（Ensemble）**：构建 `return_ensemble_size`（默认 3）个独立的 Return 网络，用集成均值作为最终输出，以降低不确定性：

$$\hat{R}(\tau) = \frac{1}{M} \sum_{m=1}^{M} f_{\varphi_m}\bigl(\mathbf{v}(\tau)\bigr)$$

---

### 2.4 偏好标签生成

Return 网络通过**成对轨迹片段的偏好比较**来训练，偏好标签由 `get_preference(pf1, pf2, ind1, ind2)` 按照如下**优先级顺序**自动生成：

```
优先级 1：通过的 PF 测试数量
  ├── sum(pf1) > sum(pf2)  → 标签 = 0（片段1更好）
  ├── sum(pf1) < sum(pf2)  → 标签 = 1（片段2更好）
  └── 相等时继续往下

优先级 2：两者均全部通过 PF 测试
  └── sum(pf1) == len(pf1)  → 标签 = 0.5（无偏好，跳过）

优先级 3：通过了哪些"更难"的 PF 测试
  └── 按 pass_count 升序排列（通过率低 = 更难）
      逐项比较，第一个有差异的项决定偏好

优先级 4：所有指标全面占优
  ├── ind1 > ind2 (全部维度)  → 标签 = 0
  └── ind1 < ind2 (全部维度)  → 标签 = 1

优先级 5：标准化后逐维比较
  └── 用 ind_rms.var 归一化，按偏度降序（ind_order）逐维比较
      第一个有差异的维度决定偏好

兜底：标签 = 0.5（无偏好）
```

**偏度排序**（`ind_order`）的意义：偏度（Skewness）高的指标在分布上更不均匀，区分力更强，因此优先用它来打破平局。

---

### 2.5 训练目标（CE Loss）

给定片段对 $(\tau_1, \tau_2)$ 及标签 $y \in \{0, 0.5, 1\}$，对第 $m$ 个 Return 网络计算 Bradley-Terry 交叉熵损失：

$$\mathcal{L}_\text{CE}^{(m)} = -\sum_{\text{batch}} \Bigl[ (1-y) \log P(\tau_1 \succ \tau_2) + y \log P(\tau_2 \succ \tau_1) \Bigr]$$

其中：

$$P(\tau_1 \succ \tau_2) = \frac{\exp\bigl(\hat{R}_m(\tau_1)\bigr)}{\exp\bigl(\hat{R}_m(\tau_1)\bigr) + \exp\bigl(\hat{R}_m(\tau_2)\bigr)} = \text{softmax}\bigl([\hat{R}_m(\tau_1),\ \hat{R}_m(\tau_2)]\bigr)[0]$$

标签 $y=0.5$ 的样本会从准确率计算中排除，但仍参与损失计算（作为等权重的软标签）。

---

### 2.6 稳定性：Change Penalty（MSE Loss）

在每轮训练前，先用上一轮的参数对所有片段计算 $\hat{R}$ 并固定（`torch.no_grad()`）作为"锚点"（Anchor）：

$$\mathcal{L}_\text{MSE}^{(m)} = \Bigl(\hat{R}_m^{\text{anchor}}(\tau_1) - \hat{R}_m(\tau_1)\Bigr)^2 + \Bigl(\hat{R}_m^{\text{anchor}}(\tau_2) - \hat{R}_m(\tau_2)\Bigr)^2$$

该损失防止 Return 网络在新一轮训练中产生剧烈漂移（catastrophic forgetting），保持输出的平滑变化。

---

### 2.7 梯度合并策略（loss_combine）

| 策略 | 说明 |
|------|------|
| `"weight"` | $\mathcal{L} = \mathcal{L}_\text{CE} + \lambda \cdot \mathcal{L}_\text{MSE}$，$\lambda$ = `change_penalty` |
| `"grad_norm"` | 将两梯度归一化到相同范数后加权合并 |
| `"weighted_grad_norm"` | 按梯度范数的倒比例动态加权 |
| `"early_stop"` | 若 $\|\nabla \mathcal{L}_\text{MSE}\| > \alpha \cdot \|\nabla \mathcal{L}_\text{CE}\|$（$\alpha$ = `es_coef`）则直接停止本轮更新；否则合并梯度 $g = g_\text{CE} + \beta \cdot g_\text{MSE}$（$\beta$ = `mse_coef`）|

**推荐使用 `"early_stop"`**：当稳定性梯度远大于偏好学习梯度时，说明本轮数据噪声过大，提前停止比强行更新更稳健。

---

## 3. Reward 网络

### 3.1 作用

Reward 网络 $r_\theta$ 回答的问题是：

> **"在状态 $s$ 下执行动作 $a$，应该获得多少即时奖励？"**

它将 Return 网络学到的"轨迹级别回报信号"**分解**到每个时间步，供 MADDPG 等算法直接使用。

---

### 3.2 输入：局部状态-动作对

**单智能体**（`TdRLRewardModel`）：

$$\text{输入} = [s, a] \in \mathbb{R}^{d_s + d_a}$$

**多智能体**（`MATdRLRewardModel`）：每个智能体 $i$ 有独立的 Reward 网络，只接收该智能体自己的局部观测和动作：

$$\text{输入}_i = [o_i, a_i] \in \mathbb{R}^{d_{o_i} + d_{a_i}}$$

这体现了**去中心化执行（Decentralized Execution）**的原则——推断时每个智能体只依赖自己的局部信息。

从联合向量中提取单个智能体的局部数据由 `_extract_agent_sa()` 完成：

```python
# 联合向量布局：[obs0, act0, obs1, act1, ..., obsN, actN]
# 对于 simple_spread_v3 (obs_dim=18, act_dim=5):
# agent 0: dims  0~22  (18+5=23 dims)
# agent 1: dims 23~45  (18+5=23 dims)
# agent 2: dims 46~68  (18+5=23 dims)
```

---

### 3.3 网络结构

与 Return 网络相同的 MLP 骨架，只是输入维度不同：

**单智能体**：
```
Linear(ds+da → 256) → LeakyReLU
Linear(256 → 256)   → LeakyReLU
Linear(256 → 256)   → LeakyReLU
Linear(256 → 1)                    ← 无激活或 Tanh
```

**多智能体**（每个智能体独立，以 agent $i$ 为例）：
```
Linear(obs_dim_i + act_dim_i → 256) → LeakyReLU   # = 23 for spread_v3
Linear(256 → 256)                   → LeakyReLU
Linear(256 → 256)                   → LeakyReLU
Linear(256 → 1)
```

同样使用集成（`ensemble_size` 个，默认 3），推断时取均值：

$$\hat{r}_i(o_i, a_i) = \frac{1}{E} \sum_{e=1}^{E} r_{\theta_i^{(e)}}(o_i, a_i)$$

每个智能体有**独立的集成 + 独立的 Adam 优化器**：

```python
self.reward_ensembles[agent_id]   # List[nn.Module], 长度 = ensemble_size
self.reward_opts[agent_id]        # torch.optim.Adam
```

---

### 3.4 训练目标（MSE Loss）

Reward 网络不直接接触偏好标签，而是将 Return 网络的输出作为监督目标：

$$\mathcal{L}_\text{reward}^{(e)} = \left\| \sum_{t=1}^{T} r_{\theta^{(e)}}(s_t, a_t)\ -\ \hat{R}_\text{target} \right\|^2$$

其中 $\hat{R}_\text{target}$ 是所有 Return 网络集成均值（固定，不回传梯度）：

$$\hat{R}_\text{target} = \frac{1}{M} \sum_{m=1}^{M} f_{\varphi_m}(\mathbf{v}(\tau))$$

约束条件是：**轨迹内所有时间步的奖励之和**等于 Return 网络预测的累积回报。这使得 Reward 网络将回报"摊销"到每个时间步，学习逐步的、步骤级别的奖励函数。

---

### 3.5 多智能体扩展：奖励分配

在全合作场景（如 `simple_spread_v3`）中，所有智能体共享同一个累积回报 $\hat{R}$。为了让每个智能体的 Reward 网络都有合理的训练目标，采用**均等分配**策略：

$$\hat{R}_{\text{target},\ i} = \frac{\hat{R}_\text{target}}{N}$$

其中 $N$ 为智能体数量。训练约束变为：

$$\sum_{t=1}^{T} r_{\theta_i^{(e)}}(o_{i,t}, a_{i,t}) \approx \frac{\hat{R}_\text{target}}{N}$$

代码实现：

```python
r_t_per_agent = r_t / self.n_agents   # (mb, 1)

for agent_id in range(self.n_agents):
    r_hat_sum = r_hat.sum(axis=1)     # sum over segment timesteps → (mb, 1)
    curr_loss = nn.MSELoss()(r_hat_sum, r_t_per_agent)
```

> **注**：均等分配是最简单的选择。若需要差异化分配（如按贡献度），可扩展为学习的分配权重 $w_i$（$\sum w_i = 1$），但这需要额外的机制来估计贡献度。

---

## 4. 两网络的协作关系

```
─────────── 训练时序 ──────────────────────────────────────────────────────────

  每 reward_update 个 episode 触发一次 _train_reward_model()
  │
  ├─ 1. prepare_training()
  │     ├── 将轨迹库切成固定长度片段（size_segment）
  │     ├── 对每段计算 ind_test 向量 → self.inds         (N_seg, K)
  │     ├── 对每段计算 pf_test 向量  → self.pfs          (N_seg, P)
  │     └── 更新 pass_count, ind_rms, ind_order（偏度排序）
  │
  ├─ 2. train_returns()  [迭代 return_train_epochs 次]
  │     ├── 随机采样片段对
  │     ├── 生成偏好标签（get_preference）
  │     ├── 计算 CE Loss + Change Penalty MSE Loss
  │     └── 按 loss_combine 策略更新 Return 网络参数
  │
  └─ 3. train_reward()   [迭代 reward_train_epochs 次]
        ├── 从 Return 网络获取目标 R̂_target（no_grad）
        ├── 计算各智能体 Reward 网络的 segment-sum MSE Loss
        └── 更新各智能体 Reward 网络参数（独立 Adam）

─────────── 推断时序 ──────────────────────────────────────────────────────────

  环境交互（每步）
  └── reward_model_ready == True 后：
      r̂_i = reward_model.r_hat(agent_id, joint_sa)
           → 替换 env reward 存入 Replay Buffer
```

---

## 5. 数据流全景

以 `simple_spread_v3`（3 智能体，`obs_dim=18`，`act_dim=5`）为例：

```
环境一步输出
  obs_list   [18, 18, 18]
  act_list   [ 5,  5,  5]
  s_next_list[18, 18, 18]
       │
       ▼  add_data()
  joint_sa    (1, 69)   = concat([obs0,act0], [obs1,act1], [obs2,act2])
  joint_s_next(1, 54)   = concat([s_next0, s_next1, s_next2])
       │
       ▼ 积累到 FIFO 轨迹库
  inputs   List[ (T, 69) ]   ← max_size=200 条轨迹
  s_nexts  List[ (T, 54) ]
       │
       ▼ prepare_training() → 切片 (size_segment=25)
  input_segments  (N_seg, 25, 69)
  s_next_segments (N_seg, 25, 54)
       │
       ├──► Tester.ind_test()
       │      输入: (N_seg, 25, 69) + (N_seg, 25, 54)
       │      输出: inds (N_seg, 3)   ← [avg_min_dist, coverage_count, collision_count]
       │
       ├──► Tester.pf_test()
       │      输出: pfs  (N_seg, 2)   ← [all_covered, no_collision]
       │
       ▼  train_returns()
  Return Network  input: (mb, 3) → output: (mb, 1)  标量 R̂
       │          CE loss + MSE(anchor) loss
       │
       ▼  train_reward()   ← target = R̂ / 3  (均等分配)
  Reward Network_0  input: (mb, 25, 23) → sum→(mb,1)  ≈ R̂/3
  Reward Network_1  input: (mb, 25, 23) → sum→(mb,1)  ≈ R̂/3
  Reward Network_2  input: (mb, 25, 23) → sum→(mb,1)  ≈ R̂/3
       │
       ▼  r_hat(agent_id, joint_sa)  每步推断
  r̂_0, r̂_1, r̂_2  → 写入 Replay Buffer 替换环境奖励
       │
       ▼  MADDPG 使用 r̂ 训练 Actor/Critic
```

---

## 6. 超参数说明

| 参数 | 含义 | 默认值 | 建议范围 |
|------|------|--------|----------|
| `return_ensemble_size` | Return 网络集成数量 | 3 | 1~5 |
| `ensemble_size` | Reward 网络集成数量（每个智能体） | 3 | 1~5 |
| `size_segment` | 轨迹切片长度 | 25 | = `max_episode_len` |
| `max_size` | FIFO 轨迹库容量（条轨迹数） | 200 | 100~500 |
| `mb_size` | 训练 minibatch 大小 | 128 | 64~512 |
| `lr` | Return/Reward 网络学习率 | 3e-4 | 1e-4~1e-3 |
| `loss_combine` | 梯度合并策略 | `"early_stop"` | 见第 2.7 节 |
| `es_coef` | early_stop 阈值 $\alpha$ | 10.0 | 5~20 |
| `mse_coef` | MSE 梯度缩放系数 $\beta$ | 1.0 | 0.1~2.0 |
| `change_penalty` | weight 策略下的 $\lambda$ | 0.0 | 0~1 |
| `reward_update` | 每隔多少 episode 训练一次 | 50 | 20~200 |
| `num_interact` | 开始训练前的预热步数 | 5000 | 1000~10000 |
| `return_train_epochs` | 每次触发时 Return 网络训练轮数 | 5 | 3~10 |
| `reward_train_epochs` | 每次触发时 Reward 网络训练轮数 | 5 | 3~10 |
| `use_kan` | 是否用 FastKAN 替换 MLP | `False` | — |
| `spectral_norm` | Return 网络是否加谱归一化 | `False` | — |

---

## 7. 单智能体 vs. 多智能体对比

| 维度 | 单智能体 (`TdRLRewardModel`) | 多智能体 (`MATdRLRewardModel`) |
|------|------------------------------|-------------------------------|
| **Reward 网络数量** | 1 组集成（共享） | N 组集成（每个智能体独立） |
| **Reward 网络输入** | $(s, a) \in \mathbb{R}^{d_s+d_a}$ | $(o_i, a_i) \in \mathbb{R}^{d_{o_i}+d_{a_i}}$ |
| **Return 网络数量** | 1 组集成（共享） | 1 组集成（所有智能体共享）|
| **Return 网络输入** | ind_test 向量 $(K,)$ | ind_test 向量 $(K,)$（联合轨迹）|
| **轨迹存储格式** | `(T, ds+da)` | `(T, N*(obs_dim+act_dim))` |
| **测试输入** | 单智能体 $(s,a)$ | 联合 `(o_0,a_0,...,o_{N-1},a_{N-1})` |
| **奖励分配** | 直接输出 $\hat{r}$ | 均等分配 $\hat{R}/N$ 作为监督目标 |
| **优化器** | 1 个 Adam（Reward）+ 1 个 Adam（Return）| N 个 Adam（Reward）+ 1 个 Adam（Return）|
| **推断接口** | `r_hat(joint_sa)` | `r_hat(agent_id, joint_sa)` |
| **CTDE 兼容性** | 不涉及 | ✅ 训练集中（联合测试），执行去中心化（局部 obs+act）|
