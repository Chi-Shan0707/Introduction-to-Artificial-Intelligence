# Q-Learning Bonus 2：CartPole 表格强化学习优化版

本项目实现了一个面向 `CartPole-v1` 的 Q-Learning 智能体。代码目标是在不修改统一评测脚本的前提下，通过优化状态离散化、探索率衰减、reward shaping 和 checkpoint 保存策略，把基础表格 Q-Learning 推向更高、更稳定的表现。

该版本对应 Bonus 2 的优化方向：允许放开基础版本的固定基座，重点提升 Q-learning 或 MCTS 在 `evaluate.py --seed-base 42 --seed-count 100 --max-episode-steps 2000` 标准口径下的平均步数。

## 1. 项目背景

CartPole 是一个经典控制问题。智能体需要通过左右推动小车，让杆子尽可能长时间保持直立。环境状态是 4 维连续向量：

| 状态变量 | 含义 |
|---|---|
| `cart_pos` | 小车位置 |
| `cart_vel` | 小车速度 |
| `pole_angle` | 杆子角度 |
| `pole_angular_velocity` | 杆子角速度 |

动作空间是离散的：

```text
0 = 向左推
1 = 向右推
```

本实现将单局最大步数设为 `2000`，用于区分普通可行策略和接近满分的长期稳定控制策略。

## 2. 文件与输出

建议将代码文件命名为：

```text
train_qlearning_bonus2.py
```

训练后会生成：

```text
checkpoints/
├── q_learning_bonus2.pkl
├── q_learning_bonus2_best.pkl
└── q_learning_bonus2_curve.png
```

其中更推荐用于评测的是：

```text
checkpoints/q_learning_bonus2_best.pkl
```

因为它不是训练结束时的最终 Q 表，而是训练过程中通过周期性贪心评估筛选出的 best-so-far checkpoint。

## 3. 核心实现

### 3.1 Q 表设计

代码使用表格型 Q-Learning，为每个离散状态和动作维护 Q 值：

```python
self.q_table = np.random.uniform(
    low=0,
    high=1,
    size=(MAX_DIGITIZE_NUM ** n_state, n_action)
)
```

本版本设置：

```python
MAX_DIGITIZE_NUM = 10
n_state = 4
n_action = 2
```

因此 Q 表大小为：

```text
10^4 × 2 = 10000 × 2
```

相比基础版本常见的 `6^4 = 1296` 个离散状态，这一版本能更细致地区分连续状态差异。

### 3.2 连续状态离散化

CartPole 的状态是连续的，而 Q 表只能索引离散状态，因此代码通过 `make_bins_state()` 将 4 维状态转换为整数 `state_id`。

核心思路是：

1. 先把每个物理变量裁剪到合理范围；
2. 再把每个变量映射到 10 个 bin；
3. 最后按位权合成为一个唯一状态编号。

本版本使用的 clip 范围如下：

| 状态变量 | 范围 | 优化意义 |
|---|---:|---|
| `cart_pos` | `[-2.4, 2.4]` | 覆盖小车位置终止边界 |
| `cart_vel` | `[-3.5, 3.5]` | 覆盖较高速运动状态 |
| `pole_angle` | `[-0.42, 0.42]` | 精确覆盖杆子角度终止边界 |
| `pole_angular_velocity` | `[-3.0, 3.0]` | 避免角速度过早饱和到边界 bin |

其中 `pole_angle` 是最关键的变量。CartPole 的实际角度终止边界约为 `±0.418`，因此把范围设为 `±0.42` 可以让全部 bin 都集中在有效控制区域内，而不是浪费在几乎不会存活的极端角度上。

### 3.3 Bellman TD 更新

Q 值更新使用标准 Bellman 时序差分公式：

```text
Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') - Q(s, a) ]
```

代码中的实现是：

```python
best_next_q = np.max(self.q_table[next_id, :])
td_target = reward + self.gamma * best_next_q
td_error = td_target - self.q_table[current_id, action]
self.q_table[current_id, action] += self.lr * td_error
```

这使智能体可以通过当前动作带来的即时反馈和下一状态的长期价值，逐步修正当前 Q 值。

### 3.4 ε-greedy 动作选择

训练阶段使用 ε-greedy：

```python
if np.random.uniform(0, 1) < self.epsilon:
    action = np.random.choice(self.n_action)
else:
    action = np.argmax(self.q_table[current_id, :])
```

这让智能体在训练早期保持探索，在训练后期逐渐转向利用已学到的最优动作。

### 3.5 Reward shaping

基础 CartPole 的 reward 是每存活一步 +1，信号相对稀疏。该代码改用终止惩罚：

```python
if terminated or truncated:
    reward = step - TARGET
else:
    reward = 0
```

含义是：

- 越早失败，惩罚越大；
- 越接近 2000 步，惩罚越小；
- 存活过程中不给额外正奖励。

这种设计把“早死很糟糕”的信息清晰地反向传播到 Q 表里，有助于智能体学习避免导致快速失败的动作。

## 4. 优化思路

### 优化一：提高状态离散化精度

基础表格 Q-Learning 的主要瓶颈是状态表示粗糙。若 bin 数太少，很多物理意义不同的状态会被映射到同一个离散格子中，导致 Q 表无法给出精细动作。

本版本将：

```text
MAX_DIGITIZE_NUM: 6 → 10
```

状态数从：

```text
1296 → 10000
```

这样能显著减少“不同状态共用同一 Q 值”的问题。代价是状态空间变大，需要更多训练轮数和更合理的探索策略。

### 优化二：重设 clip 范围，让 bin 用在有效区域

增加 bin 数并不一定有效。如果 clip 范围过宽，很多 bin 会被浪费在环境很快终止的区域。该代码针对 CartPole 的真实终止边界重新设置了范围，尤其是把 `pole_angle` 收紧到 `±0.42`，提高关键角度区域的分辨率。

这相当于把有限的 Q 表容量集中用于真正影响控制决策的状态区域。

### 优化三：指数衰减探索率

本版本使用：

```python
EPSILON = 0.5
EPSILON_MIN = 0.005
EPSILON_DECAY = 0.9993
```

训练后每一轮执行：

```python
epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
```

优化逻辑是：

1. 前期高 ε，充分探索 10000 个离散状态；
2. 中期逐渐转向利用已有 Q 值；
3. 后期保留极小探索，避免完全固化。

这比固定 ε 更稳定，尤其适合状态空间变大的 Bonus 2 版本。

### 优化四：增加训练轮数与学习率

由于状态数大幅增加，基础版本训练轮数不足以覆盖足够多状态。因此本版本设置：

```python
EPOCHS = 6000
LR = 0.08
```

更多训练轮数用于覆盖状态空间，更高学习率用于加快 Q 值修正。

### 优化五：周期性贪心评估与 best checkpoint

Q-Learning 训练曲线往往会震荡。训练结束时的模型不一定是训练过程中最好的模型。因此代码每隔一段时间进行一次纯贪心评估：

```python
EVAL_INTERVAL = 300
EVAL_EPISODES = 20
```

如果当前 Q 表的贪心平均步数刷新历史最好值，就保存为：

```text
q_learning_bonus2_best.pkl
```

这项优化非常关键，因为它把最终提交结果从“训练最后一刻”改为“训练全过程中表现最好的时刻”。

### 优化六：训练曲线记录

代码保存训练曲线，并绘制 50 episode 滑动均值：

```text
q_learning_bonus2_curve.png
```

该图可以用于观察训练是否持续提升、是否进入平台期，以及后期是否发生策略退化。

## 5. 运行方式

### 训练

```bash
python train_qlearning_bonus2.py
```

### 标准评测

```bash
python evaluate.py \
    --agent-class train_qlearning_bonus2:Agent \
    --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.08,"gamma":0.99,"epsilon":0.0}' \
    --checkpoint checkpoints/q_learning_bonus2_best.pkl \
    --seed-base 42 \
    --seed-count 100 \
    --max-episode-steps 2000
```

评测时建议使用 `epsilon=0.0`，即完全贪心策略。

### 可视化

```bash
python vis.py \
    --agent-class train_qlearning_bonus2:Agent \
    --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.08,"gamma":0.99,"epsilon":0.0}' \
    --checkpoint checkpoints/q_learning_bonus2_best.pkl \
    --seed 42 \
    --output checkpoints/q_learning_bonus2_seed42.gif
```

## 6. 与基础版本的对比

| 模块 | 基础版本 | 本优化版 | 优化目的 |
|---|---:|---:|---|
| 每维 bin 数 | 6 | 10 | 提高状态分辨率 |
| 离散状态数 | 1296 | 10000 | 减少状态混淆 |
| 初始 ε | 较低或固定 | 0.5 | 前期充分探索 |
| 最小 ε | 无或较高 | 0.005 | 后期稳定利用 |
| ε 调度 | 固定 / 简单衰减 | 指数衰减 | 平衡探索和利用 |
| 训练轮数 | 约 2000 | 6000 | 适应更大状态空间 |
| 学习率 | 0.04 | 0.08 | 加快 Q 值更新 |
| 模型保存 | 保存最终版本 | 保存 best-so-far | 避免训练末期震荡 |

## 7. 总结

该 Q-Learning Bonus 2 版本的核心优化可以概括为三点：

1. **让状态表达更细**：通过 `10^4` 状态和精确 clip 范围提升 Q 表表达能力；
2. **让探索更合理**：通过指数衰减 ε 先探索再利用；
3. **让最终模型更稳**：通过周期性贪心评估保存 best-so-far checkpoint。

整体思路不是单纯增加训练时间，而是围绕表格 Q-Learning 的状态表示、探索机制和模型选择机制进行系统优化。
