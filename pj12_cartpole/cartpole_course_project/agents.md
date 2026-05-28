# Agents 详细说明

本文档详细记录 CartPole 项目中所有 Agent 的架构、参数、接口和行为。

---

## 1. 环境与评估标准

**环境**: `gymnasium.CartPole-v1`
- 状态: 4 维连续 `[cart_pos, cart_vel, pole_angle, pole_angular_vel]`
- 动作: `0=推左, 1=推右`
- 终止条件: `|x|>2.4` 或 `|θ|>0.418` 或步数到上限
- 每步 reward=+1，累计回报=存活步数

**评估** (`evaluate.py`):
- `max_episode_steps=2000`，100 个 seed (42~141)
- `perturb_scale`: 初始状态加高斯扰动 std
- 评估时 `epsilon` 强制 0

**统一接口**:
```python
class Agent:
    def __init__(**init_kwargs)
    def load_model(checkpoint_path)
    def predict(state) -> int          # evaluate/vis 用
    def policy_info(state) -> dict     # vis 右上角叠加
    def save_model(filepath)
```

---

## 2. 基础 Q-Learning Agent

**文件**: `train_qlearning.py`
**Checkpoint**: `checkpoints/q_learning_model.pkl`

### 架构

- 状态离散化: 4 维连续 → 6 bins/维 → 6⁴ = **1296 离散状态**
- Q 表: `shape=(1296, 2)`，随机初始化 `[0, 1)`
- 查表: `state_id = s1 + s2×6 + s3×36 + s4×216`

### 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| LR | 0.04 | 学习率 |
| GAMMA | 0.99 | 折扣因子 |
| EPSILON | 0.1 | 探索率 (训练时) |
| EPOCHS | 2000 | 训练轮数 |
| TARGET | 1000 | 训练内层 TimeLimit |
| MAX_DIGITIZE_NUM | 6 | 每维 bin 数 |

### 离散化

```python
CLIP_RANGES = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]

def make_bins_state(self, current_state):
    # 每维 clip → np.argwhere 找 bin → 编码为 state_id
    return s1 + s2*6 + s3*36 + s4*216
```

### 核心方法

```python
# TODO 1: Bellman TD 更新
def update_q_table(self, current_state, action, reward, next_state):
    td_target = reward + gamma * max(Q[next_state])
    Q[current_state, action] += lr * (td_target - Q[current_state, action])

# TODO 2: ε-greedy
def decide_action(self, current_state):
    if random() < epsilon: return random_action()
    else: return argmax(Q[current_state])

# TODO 3: reward shaping
if terminated or truncated:
    reward = step - TARGET   # 距目标越远惩罚越大
```

### 评测结果

| 指标 | perturb=0 | perturb=0.05 |
|------|-----------|--------------|
| mean | 205.7 | 194.1 |
| median | 198 | 192 |
| std | 31.4 | 45.5 |
| max_ratio | 0% | 0% |

---

## 3. REINFORCE Agent

**文件**: `train_reinforce.py`
**Checkpoint**: `checkpoints/reinforce_model.pt`

### 架构

```
输入 (4) → Linear(4, 16) → ReLU → Linear(16, 2) → Softmax → 动作概率
```

### 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| LR | 0.005 | 学习率 |
| GAMMA | 0.99 | 折扣因子 |
| HIDDEN_C | 16 | 隐藏层宽度 |
| MAX_T | 2000 | 单 episode 最大步数 |

### 核心方法

```python
# TODO 1: 前向传播
def forward(self, state):
    x = relu(fc1(state))
    return softmax(fc2(x), dim=1)

# TODO 2: 逐步折扣回报 (从后向前)
G_t = r_t + gamma * G_{t+1}

# TODO 3: returns 标准化 (减方差技巧)
returns = (returns - returns.mean()) / (returns.std() + 1e-9)

# TODO 4: policy loss
loss = -(log_probs * returns).sum()
```

### 评测结果

| 指标 | perturb=0 |
|------|-----------|
| mean | **2000.0** |
| median | 2000 |
| std | 0.0 |
| max_ratio | **100%** |

---

## 4. MCTS Agent

**文件**: `train_mcts.py`
**Checkpoint**: 无 (在线规划)

### 核心机制

每步决策在线搭搜索树: 选择 → 扩展 → 随机模拟 → 回溯

```
UCT(child) = Q/N + C_p × √(2 × ln(N_parent) / N_child)
              ↑              ↑
           经验平均       探索奖励
```

### 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| iteration_budget | 80~100 | 每步搜索迭代数 |
| lookahead_target | 200 | 目标搜索深度 |
| C_p 初始 | 200 | 探索系数 (自适应) |

### C_p 自适应

```python
# 每轮搜索后:
if search_depth < lookahead_target:
    C_p -= 1   # 搜索太浅，减少探索
else:
    C_p += 1   # 搜索够深，鼓励探索
```

### 评测结果

| 指标 | perturb=0 |
|------|-----------|
| mean | 233.1 |
| median | 231 |
| std | 11.6 |
| max_ratio | 0% |

---

## 5. Bonus 2: 优化 Q-Learning Agent

**文件**: `train_qlearning_bonus2.py`
**Checkpoint**: `checkpoints/q_learning_bonus2_model.pkl`

### 与基础版的差异

| 参数 | 基础版 | 优化版 |
|------|--------|--------|
| MAX_DIGITIZE_NUM | 6 (1296 状态) | **12 (20736 状态)** |
| EPSILON | 固定 0.1 | **1.0 → 0.01 指数衰减** (×0.99975) |
| EPOCHS | 2000 | **15000** |
| TARGET | 1000 | **2000** |
| Q 表初始化 | random [0,1) | **zeros** |
| 奖励 | 稀疏 (仅结束给) | **稠密 (每步 +1)** |
| 最佳保存 | 无 | **best-so-far** |

### ε 衰减

```python
epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
# 1.0 → 0.01，约 18000 步收敛
```

### 评测结果

| 指标 | 数值 |
|------|------|
| mean | **1790.6** |
| median | **2000** |
| std | 504.6 |
| max_ratio | **81%** |

---

## 6. Bonus 3: Smooth Agent (邻域平滑) — 失败

**文件**: `train_qlearning_bonus3_smooth.py`
**Checkpoint**: `checkpoints/q_learning_bonus3_smooth_model.pkl`

### 架构

与基础 Q-Learning 相同 (6 bins, 1296 状态)，但**决策时**用邻域插值 Q 值：

```python
def _smooth_q_values(self, current_state):
    # 4D multilinear 插值: 对 2^4=16 个相邻 bin 加权
    for offset_bits in range(16):
        w = 1.0
        for dim in range(4):
            if (offset_bits >> dim) & 1:
                w *= frac[dim]          # 右邻居权重 = frac
                neighbor = bin_idx + 1
            else:
                w *= (1 - frac[dim])    # 左邻居权重 = 1-frac
                neighbor = bin_idx
        q_vals += w * Q[neighbor_state_id]
    return q_vals
```

### 失败原因

**行为策略 (平滑) ≠ 学习目标 (离散)**:
- `decide_action` 用插值 Q 值选动作 → 行为策略是平滑的
- `update_q_table` 只更新离散 bin 的 Q 值 → 学习目标是离散的
- 平滑 Q 在 bin 边界混合邻居值，可能选出与离散 Q 不同的动作
- Q 表更新方向被干扰，训练不稳定

### 评测结果

| 指标 | perturb=0 | perturb=0.05 |
|------|-----------|--------------|
| mean | **113.2** | **108.4** |
| vs 基线 | **-44.1%** | **-44.2%** |
| 状态 | 失败 | 失败 |

---

## 7. Bonus 3: Hedge Agent (多表集成) — 失败

**文件**: `train_qlearning_bonus3_hedge_v2.py`
**Checkpoint**: `checkpoints/q_learning_bonus3_hedge_v2_agent.pkl`

### 架构

**Phase 1**: 训练 4 张独立 Q 表，bin 边界偏移不同：

| 表 | offset | 含义 |
|----|--------|------|
| Table 0 | 0.00 | 无偏移 |
| Table 1 | 0.25 | 偏移 0.25 bin 宽 |
| Table 2 | 0.50 | 偏移 0.50 bin 宽 |
| Table 3 | 0.75 | 偏移 0.75 bin 宽 |

每张表独立训练 2000 epochs，训练后**冻结**。

**Phase 2**: Hedge 在线加权集成

```python
# Hedge 更新 (Freund & Schapire):
w_i^{t+1} = w_i^t × exp(-η × loss_i)
loss_i = 1 - softmax(Q_i(s))[action_taken]
# 归一化: w /= sum(w)
```

### 失败原因 — 权重坍缩

```
Final weights: [5e-324, 5e-324, 1.0, 5e-324]
```

Hedge 的指数更新导致正反馈：
1. 某张表在早期 episode 略好 → 权重指数增长
2. 更高权重 → 更多动作来自该表 → 该表 loss 更低 → 权重继续增长
3. 其他表权重指数衰减至 0
4. 最终只剩 1 张表，集成效果完全消失

而 Table 2 (被选中的表) 训练末期仅 127 步 (最差的一张)。

### 评测结果

| 指标 | perturb=0 | perturb=0.05 |
|------|-----------|--------------|
| mean | **169.1** | **165.3** |
| vs 基线 | **-14.8%** | **-14.8%** |
| 状态 | 失败 | 失败 |

---

## 8. Bonus 3: AdaptiveGrad Agent (梯度驱动自适应) — 唯一成功

**文件**: `train_qlearning_bonus3_adaptive_gradient.py`
**Checkpoint**: `checkpoints/q_learning_bonus3_adaptive_gradient_model.pkl`

### 核心创新

**同样是 6 bins/维、1296 状态**，但 bin 边界从均匀变为按 Q 值梯度分布。

```
均匀 (baseline):
  cart_pos: [-2.4 ── -1.6 ── -0.8 ── 0.0 ── 0.8 ── 1.6 ── 2.4]
  平衡点附近 (|x|<0.8) 只有 2 个 bin

AdaptiveGrad:
  cart_pos: [-2.4 ──────── -0.3 ── -0.1 ── 0.1 ── 0.3 ──────── 2.4]
  平衡点附近 (|x|<0.3) 有 4 个 bin，分辨率 ×4
```

### 超参数

| 参数 | 值 | 说明 |
|------|------|------|
| LR | 0.04 | 学习率 |
| GAMMA | 0.99 | 折扣因子 |
| EPSILON | 0.1 | 探索率 |
| EPOCHS | 2000 | 训练轮数 |
| MAX_DIGITIZE_NUM | 6 | 每维 bin 数 (不变) |
| ADAPT_INTERVAL | 250 | 边界调整间隔 (episodes) |
| GRAD_ALPHA | 15.0 | 梯度密度系数 |

### 自适应流程

```
Episode 0-249:   均匀边界训练 → 学习基本 Q 值
Episode 250:     计算 Q 值梯度 → 重新分配边界 → Q 表迁移
Episode 250-499: 新边界训练
Episode 500:     再次调整
...
Episode 1750:    最后一次调整
共 7 次渐进优化
```

### Step 1: 计算密度

```python
def _adapt_boundaries(self):
    for dim in range(4):
        # 每 bin 的平均 max-Q
        q_per_bin = [mean(max_Q(states_in_bin)) for b in range(6)]
        
        # 相邻 bin 的梯度
        gradients = [abs(q_per_bin[b+1] - q_per_bin[b]) + 0.005 
                     for b in range(5)]
        
        # 密度 = 1 + α × gradient
        # α=15 → 梯度大的区域密度可达 16 倍
        dens = 1.0 + GRAD_ALPHA * gradients[b]
```

### Step 2: 重采样边界

```python
        # 累积密度 → 均匀分位点 → 新边界位置
        cum_density = [0.0]
        for b in range(6):
            width = boundaries[b+1] - boundaries[b]
            cum_density.append(cum_density[-1] + dens * width)
        
        # 在累积密度曲线上均匀放置 5 个内部分界点
        for i in range(1, 6):
            target = total_dens * i / 6
            # 线性插值找位置
```

### Step 3: Q 表迁移

```python
def _remap_q_table(self, old_boundaries, momentum=0.3):
    # 对每个新 bin 中心，找到旧边界下对应的 bin
    # 最近邻映射 + 动量平滑
    self.q_table = 0.3 * old_q + 0.7 * mapped_q
```

### 为什么比 AdaptiveTD 好

| 对比 | AdaptiveGrad | AdaptiveTD |
|------|-------------|------------|
| 信号 | Q 值**空间**梯度 (相邻 bin 之差) | TD 误差**时间**残差 (单步更新) |
| 稳定性 | 稳定 — 反映 Q 函数结构 | 噪声大 — 反映训练状态 |
| 含义 | "Q 值变化快 → 需要更多分辨率" | "Q 值没收敛" (≠ 需要更多 bin) |
| 结果 | mean=1906 (82% 达上限) | mean=145 (比基线低 29%) |

### 评测结果

| 指标 | perturb=0 | perturb=0.05 |
|------|-----------|--------------|
| mean | **1906.0** | **1865.8** |
| median | 2000 | 2000 |
| std | 244.2 | 315.1 |
| max_ratio | **82%** | **79%** |
| vs 基线 | **+826.8%** | **+861.3%** |
| 状态 | **成功** | **成功** |

---

## 9. Bonus 3: AdaptiveTD Agent (TD 误差驱动) — 失败

**文件**: `train_qlearning_bonus3_adaptive_tderror.py`
**Checkpoint**: `checkpoints/q_learning_bonus3_adaptive_tderror_model.pkl`

### 与 AdaptiveGrad 的唯一区别

密度信号从 Q 值梯度改为 TD 误差：

```python
# 训练时记录 TD 误差 (指数移动平均)
self.td_errors[current_id] = 0.9 * self.td_errors[current_id] + 0.1 * abs(td_error)

# 调整时密度 = 1 + TD_ALPHA × td_error_per_bin
dens = 1.0 + TD_ALPHA * td_per_bin[b]   # TD_ALPHA=25
```

### 失败原因

1. **信号噪声大**: 训练初期 TD 误差普遍很大，无法区分"需要更多分辨率"和"只是噪声"
2. **方向错误**: 高 TD 误差 ≠ 需要更多 bin。TD 误差大可能只是因为该区域尚未收敛
3. **频繁打断**: 每 250 episode 调整边界 + Q 表迁移，学习被反复重置

### 评测结果

| 指标 | perturb=0 | perturb=0.05 |
|------|-----------|--------------|
| mean | **145.6** | **127.2** |
| vs 基线 | **-29.2%** | **-34.5%** |
| 状态 | 失败 | 失败 |

---

## 10. 文件索引

| 文件 | 说明 | 状态 |
|------|------|------|
| `train_qlearning.py` | 基础 Q-Learning | 必做 |
| `train_reinforce.py` | REINFORCE 策略梯度 | 必做 |
| `train_mcts.py` | MCTS 树搜索 | 必做 |
| `train_qlearning_bonus2.py` | 优化 Q-Learning | Bonus 2 |
| `train_qlearning_bonus3_smooth.py` | 邻域平滑 | Bonus 3 (失败) |
| `train_qlearning_bonus3_hedge_v2.py` | Hedge 集成 | Bonus 3 (失败) |
| `train_qlearning_bonus3_adaptive_gradient.py` | 梯度驱动自适应 | **Bonus 3 (成功)** |
| `train_qlearning_bonus3_adaptive_tderror.py` | TD 误差驱动自适应 | Bonus 3 (失败) |
| `evaluate.py` | 通用评估入口 | 禁改 |
| `vis.py` / `vis_mcts.py` | 可视化 | 禁改 |
| `bonus3_analysis.py` | 鲁棒性分析 | 已修正 |
| `verify_hedge.py` | Hedge 验证脚本 | 新增 |
