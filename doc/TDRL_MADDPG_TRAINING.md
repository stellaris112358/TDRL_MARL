# TDRL-MADDPG 训练流程详解

> 适用入口：MADDPG/train_tdrl.py

---

## 1. 目标与核心思路

TDRL-MADDPG 在 MADDPG 的集中式训练流程中引入 TdRL 奖励学习：

- 使用 Tester 对多智能体联合轨迹做 PF/指示性测试；
- 训练共享 Return Network 预测轨迹回报；
- 训练每个智能体独立 Reward Network 预测即时奖励；
- 用预测奖励替换环境奖励写入 buffer（可选）。

---

## 2. 数据流与训练循环总览

训练主循环发生在 `TdRLRunner.run()`，核心数据流如下：

1. 环境交互：MADDPG 产生动作，环境返回 `(s_next, r, done)`
2. 轨迹缓存：调用 `MATdRLRewardModel.add_data()` 存储 joint 轨迹
3. 选择写入 buffer 的奖励：
   - 若 reward model 已训练完成且 `relabel_buffer=true`，使用预测奖励
   - 否则使用环境奖励
4. MADDPG 更新：从 buffer 采样，更新 Actor/Critic
5. 每隔若干 episode 训练一次 reward model（Return + Reward）

---

## 3. 训练流程分解（按时间顺序）

### 3.1 初始化阶段

- 创建 MPE 环境（pettingzoo wrapper）
- 初始化 MADDPG agent + replay buffer
- 初始化 `MATdRLRewardModel`
  - 每个智能体独立 Reward Network
  - 所有智能体共享 Return Network
  - Tester 使用 joint 轨迹做 PF/ind 测试

### 3.2 每步交互与轨迹收集

每个 step：

1. MADDPG 选择动作（ε-贪心 + 高斯噪声）
2. 环境执行并返回 joint reward
3. 调用 `reward_model.add_data(...)` 写入 joint 轨迹数据
4. 若已训练好 reward model 且开启重标注：
   - 构造 joint_sa
   - 调用 `reward_model.r_hat(agent_id, joint_sa)` 获取每个智能体预测奖励
   - 用预测奖励写入 MADDPG buffer

说明：MADDPG 的训练更新仍在每个 step 后进行，只有奖励来源可被替换。

### 3.3 TdRL 奖励模型训练触发

在每个 episode 结束时触发检查：

- 如果 `time_step >= num_interact` 且
- `episode_count % reward_update == 0`

则进入 reward model 的完整训练流程。

---

## 4. Reward Model 训练细节

训练发生在 `TdRLRunner._train_reward_model()`：

### 4.1 轨迹准备

- `reward_model.prepare_training()`
  - 将轨迹按 `size_segment` 切分
  - 运行 Tester 计算 PF/ind 值
  - 维护 PF 通过率、ind 归一化统计

### 4.2 Return Network 训练

- 目标：学习 “测试指标 -> 轨迹回报”
- `train_returns()`：
  - 生成轨迹对的偏好标签
  - 交叉熵损失 + 变化惩罚 MSE 损失
  - 按 `return_train_epochs` 训练

### 4.3 Reward Network 训练

- 每个智能体一套 Reward Network
- `train_reward()`：
  - Return Network 输出作为回报监督
  - MSE 损失训练
  - 按 `reward_train_epochs` 训练

### 4.4 Buffer 重标注（可选）

若 `relabel_buffer=true`：

- 遍历 buffer 中的 joint_sa
- 用 `r_hat_batch()` 重新计算奖励
- 覆盖原始奖励，后续 MADDPG 更新使用预测奖励

---

## 5. 关键配置项（MADDPG/conf/config_tdrl.yaml）

- `tdrl.reward_update`: 每 N 个 episode 训练一次 reward model
- `tdrl.num_interact`: 训练 reward model 之前的预热步数
- `tdrl.return_train_epochs`: Return Network 训练轮数
- `tdrl.reward_train_epochs`: Reward Network 训练轮数
- `tdrl.relabel_buffer`: 是否重标注 replay buffer

---

## 6. 典型命令行

```shell
cd MADDPG
python train_tdrl.py
python train_tdrl.py time_steps=1000000 tdrl.reward_update=100
python train_tdrl.py evaluate=true
```

---

## 7. 训练输出与日志

- Hydra 输出：`MADDPG/outputs/YYYY-MM-DD/HH-MM-SS/`
- 模型保存：`MADDPG/model_tdrl/`
- TensorBoard：训练日志存入 `save_dir` 对应路径

---

## 8. 常见调参建议

- 更稳定训练：调小 `tdrl.reward_update`（更频繁更新）
- 更快启动：调小 `tdrl.num_interact`
- 更平滑奖励：适当增大 `tdrl.mse_coef` 或开启 `spectral_norm`
- 任务更复杂时：增大 `return_train_epochs` 和 `reward_train_epochs`

---

## 9. TensorBoard 记录内容

TensorBoard 日志写入路径：

- `<save_dir>/<scenario_name>/tb_logs/`

可记录的标量如下：

- `loss/actor_agent{i}`：每个智能体 Actor loss（step 粒度）
- `loss/critic_agent{i}`：每个智能体 Critic loss（step 粒度）
- `reward/mean_episode_return`：每个 episode 平均回报
- `reward/agent{i}_episode_return`：每个 episode 的单个智能体回报
- `eval/mean_return`：评估模式平均回报（按 evaluate 周期）
- `explore/noise`：探索噪声强度（每 500 step 记录）
- `explore/epsilon`：探索 ε（每 500 step 记录）

TdRL 专用指标（reward model 训练时记录）：

- `tdrl/pf_mean`：PF 测试平均通过率
- `tdrl/ind_{i}`：第 i 个指示性测试统计值
- `tdrl/return_acc`：Return Network 偏好预测准确率
- `tdrl/return_loss`：Return Network 训练损失
- `tdrl/reward_loss_agent{i}`：第 i 个智能体 Reward Network 训练损失
