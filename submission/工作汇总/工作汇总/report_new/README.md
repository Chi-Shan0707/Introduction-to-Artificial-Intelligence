# CartPole 强化学习课程项目报告

## 一、评估环境与评分逻辑

**环境**: `gymnasium.CartPole-v1`
- 状态空间: 4 维连续 (cart 位置、cart 速度、pole 角度、pole 角速度)
- 动作空间: 2 维离散 (左推/右推)
- 每存活 1 步 reward = +1，累计回报 = 存活步数

**评估设置** (`evaluate.py`):
- `max_episode_steps = 2000` (覆盖默认 500)
- 100 个 seed (42~141)，初始状态可加高斯扰动 `perturb_scale`
- 评估时 epsilon 强制 0 (greedy)

**关键指标**: mean (平均步数)、median、std、reached_max_ratio (达 2000 上限比例)

---

## 二、Bonus 1: 三种基础算法对比

| 算法 | mean | median | std | max_ratio |
|------|------|--------|-----|-----------|
| **Q-learning** | 205.7 | 198 | 31.4 | 0% |
| **REINFORCE** | **2000.0** | **2000** | **0.0** | **100%** |
| **MCTS** | 232.5 | 230 | 13.7 | 0% |

REINFORCE 以策略梯度直接拟合连续状态，完美达到上限。Q-learning 受限于 6⁴=1296 状态离散化。MCTS 受搜索深度限制。

---

## 三、Bonus 2: Q-learning 性能优化

| 改进项 | 原版 | 优化后 |
|--------|------|--------|
| 离散化 bins | 6 (1296 状态) | **12** (20736 状态) |
| epsilon | 固定 0.1 | **1.0 → 0.01 指数衰减** |
| 训练轮数 | 2000 | **15000** |
| TARGET | 1000 | **2000** |
| 奖励 | 稀疏 (仅结束给) | **稠密 (每步 +1，失败惩罚)** |

| 指标 | 数值 |
|------|------|
| mean | **1790.6** |
| median | **2000** |
| std | 504.6 |
| max_ratio | **81%** |

---

## 四、Bonus 3: 鲁棒性改进

### 4.0 方法概览与结果

在 Q-learning (6 bins, 2000 epochs) 基础上，测试了 4 种改进方法在 perturb=0 和 perturb=0.05 下的表现。**所有数据均来自 `evaluate.py` 100-seed 评测，已通过 JSON 文件验证。**

| 方法 | perturb=0 | perturb=0.05 | vs 基线 | 30% 门槛 | 状态 |
|------|-----------|--------------|---------|----------|------|
| **Baseline** | 205.7 | 194.1 | — | — | — |
| **Smooth** (邻域线性插值) | 113.2 | 108.4 | **-44.1%** | FAIL | 失败 |
| **Hedge** (多表集成) | 169.1 | 165.3 | **-14.8%** | FAIL | 失败 |
| **AdaptiveGrad** (梯度驱动自适应) | **1906.0** | **1865.8** | **+861.3%** | **PASS** | **成功** |
| **AdaptiveTD** (TD 误差驱动自适应) | 145.6 | 127.2 | **-34.5%** | FAIL | 失败 |

**结论**: 只有 AdaptiveGrad (梯度驱动自适应离散化) 真正有效，其余三种方法均失败。

---

### 4.1 邻域平滑 (Smooth) — 失败

**原理**: 决策时不只用当前 bin 的 Q 值，而是对周围 16 个相邻 bin 做线性插值 (4D multilinear)，使策略表面跨 bin 边界连续。

**代码** (`train_qlearning_bonus3_smooth.py`):
```python
def _smooth_q_values(self, current_state):
    fractions = self._get_bin_fractions(current_state)
    q_vals = np.zeros(self.n_action)
    for offset_bits in range(16):  # 2^4 个邻居
        w = 1.0
        bin_indices = []
        for dim in range(4):
            bin_idx, frac = fractions[dim]
            if (offset_bits >> dim) & 1:
                neighbor_idx = min(bin_idx + 1, MAX_DIGITIZE_NUM - 1)
                w *= frac
            else:
                neighbor_idx = bin_idx
                w *= (1.0 - frac)
            bin_indices.append(neighbor_idx)
        state_id = ...
        q_vals += w * self.q_table[state_id, :]
    return q_vals
```

**失败原因**:
- **行为策略与学习目标不匹配**: `decide_action` 用插值后的平滑 Q 值选动作，但 `update_q_table` 只更新离散 bin 中心的 Q 值
- 这构成 off-policy 学习：行为策略 (平滑) ≠ 目标策略 (离散贪心)
- 平滑 Q 值在 bin 边界处混合了邻居的 Q 值，可能选出与离散 Q 不同的动作，导致 Q 表更新方向混乱
- 训练后期性能持续下降 (最后 20 轮 avg=92，基线为 144)

**实测**: perturb=0 仅 113.2 步，比基线低 44%。smoothstep (C1) 版本更差 (17.2 步)。

---

### 4.2 Hedge 多表集成 (Hedge) — 失败

**原理**: 训练 4 张独立 Q 表，每张 bin 边界有不同偏移 (0/0.25/0.5/0.75 bin 宽度)。用 Hedge 在线加权集成。

**Hedge 更新** (`train_qlearning_bonus3_hedge_v2.py`):
```python
def update_weights(self, state, action_taken):
    for i, table in enumerate(self.tables):
        probs = table.get_probs(state, temperature=1.0)
        losses[i] = 1.0 - probs[action_taken]
    self.hedge_weights *= np.exp(-self.eta * losses)
    self.hedge_weights /= np.sum(self.hedge_weights)
```

**失败原因 — 权重坍缩**:
```
Final Hedge weights: [5e-324, 5e-324, 1.0, 5e-324]
```
- Hedge 的指数更新导致正反馈循环：某张表早期略好 → 权重指数增长 → 进一步主导 → 其他表权重归零
- 4 张表的 offset 差异 (0/0.25/0.5/0.75) 不够大，缺乏真正的"多样性"
- 最终只剩 Table 2 (offset=0.50) 有非零权重，等价于单表，集成效果完全消失
- 而 Table 2 训练末期仅 127 步 (最差的一张表)，Hedge 选中了最差的表

**实测**: perturb=0 仅 169.1 步，比基线低 15%。

---

### 4.3 梯度驱动自适应离散化 (AdaptiveGrad) — 唯一成功方法

**核心思想**: 保持 **6 bins/维、1296 状态总数不变**，但把 bin 边界从均匀分布改为**按 Q 值梯度分布**——Q 值变化剧烈的地方 bin 更密，变化平缓的地方 bin 更疏。

**为什么同样 6 bins 能从 205 步跳到 1906 步？**

关键在于 CartPole 的 Q 值分布极不均匀。均匀离散化在"杆子快倒了"的区域（极端状态）和"杆子平衡了"的区域（中心状态）给了相同的分辨率，但中心区域才是决定成败的关键：

```
均匀离散化 (baseline):
  cart_pos: [-2.4 ── -1.6 ── -0.8 ── 0.0 ── 0.8 ── 1.6 ── 2.4]
  每个 bin 宽 0.8，中心和边缘分辨率相同
  → 平衡点附近 (|x|<0.8) 只有 2 个 bin，分辨率不足

AdaptiveGrad 自适应:
  cart_pos: [-2.4 ─────── -0.3 ── -0.1 ── 0.1 ── 0.3 ─────── 2.4]
  平衡点附近 bin 密集，边缘 bin 稀疏
  → 平衡点附近 (|x|<0.3) 有 4 个 bin，分辨率 ×4
```

**自适应边界的工作流程**:

```
Episode 0-249:   用均匀边界训练，学习基本 Q 值
Episode 250:     计算 Q 值梯度 → 调整边界 → Q 表迁移
Episode 250-499: 用新边界继续训练
Episode 500:     再次调整边界
...
Episode 1750:    最后一次调整
共 7 次渐进优化
```

**Step 1 — 计算密度**: 对每个维度，计算相邻 bin 之间的 Q 值梯度
```python
def _adapt_boundaries(self):
    for dim in range(4):
        # 每 bin 的平均 max-Q
        q_per_bin = [mean(max_Q(states_in_bin)) for b in range(6)]
        
        # 相邻 bin 的梯度 = |Q(b+1) - Q(b)|
        gradients = [abs(q_per_bin[b+1] - q_per_bin[b]) + 0.005 
                     for b in range(5)]
        
        # 密度 = 1 + α × gradient
        # α=15 意味着梯度大的区域密度可达 16 倍
        dens = 1.0 + GRAD_ALPHA * gradients[b]
```

**Step 2 — 重采样边界**: 在累积密度曲线上均匀放置 5 个内部分界点
```python
        # 累积密度 → 均匀分位点 → 新边界位置
        # 等价于: 在 Q 值变化剧烈的区域"挤"更多边界
```

**Step 3 — Q 表迁移**: 边界变了，但学习成果不能丢
```python
def _remap_q_table(self, old_boundaries, momentum=0.3):
    # 对每个新 bin 中心，找到旧边界下对应的 bin
    # 最近邻映射 + 动量平滑 (保留 30% 旧值)
    self.q_table = 0.3 * old_q + 0.7 * mapped_q
```

**为什么比 TD 误差驱动 (AdaptiveTD) 好？**

| 对比 | AdaptiveGrad | AdaptiveTD |
|------|-------------|------------|
| 信号来源 | Q 值在**空间维度**的梯度 (相邻 bin 之差) | TD 误差在**时间维度**的残差 (单步更新) |
| 信号稳定性 | 稳定——梯度反映 Q 函数的**结构性** | 噪声大——TD 误差反映**当前训练状态** |
| 物理含义 | "Q 值在这里变化快，需要更多分辨率" | "这里的 Q 值还没收敛" (不等于需要更多 bin) |
| 结果 | mean=1906 (82% 达上限) | mean=145 (比基线低 29%) |

**实测**:
- perturb=0: mean=1906.0 (82% 达上限)，median=2000
- perturb=0.05: mean=1865.8 (79% 达上限)，median=2000
- **同样的 1296 个状态格子，相对基线提升 +861.3%**

---

### 4.4 TD 误差驱动自适应离散化 (AdaptiveTD) — 失败

**原理**: 与 AdaptiveGrad 类似，但用累积 TD 误差代替 Q 值梯度作为密度函数。

**代码** (`train_qlearning_bonus3_adaptive_tderror.py`):
```python
# 更新时记录 TD 误差 (指数移动平均)
self.td_errors[current_id] = 0.9 * self.td_errors[current_id] + 0.1 * abs(td_error)

# 调整时密度 = 1 + TD_ALPHA × td_error_per_bin
```

**失败原因**:
1. **信号噪声大**: 训练初期 TD 误差普遍很大且不稳定，无法区分"需要更多分辨率"和"只是噪声"
2. **方向错误**: 高 TD 误差区域不一定是需要更高分辨率的区域。TD 误差大可能只是因为该区域尚未收敛，也可能是因为 Q 值本身就在快速变化
3. **频繁打断**: 每 250 episode 调整边界 + Q 表迁移，学习过程被反复重置
4. **与 AdaptiveGrad 的关键区别**: 梯度是**空间维度**上相邻 bin 的 Q 值差分，是稳定的结构性信号；TD 误差是**时间维度**上单步更新的残差，是噪声信号

**实测**: perturb=0 仅 145.6 步，比基线低 29%。

---

## 五、文件说明

```
report/
├── README.md              # 本报告
├── figures/               # 关键图表
│   ├── bonus1_boxplot.png           # 三算法对比箱线图
│   ├── bonus3_robustness_curve.png  # 鲁棒性曲线
│   ├── bonus3_boxplots.png          # 各方法箱线图
│   ├── q_learning_training_curve.png
│   ├── reinforce_training_curve.png
│   ├── q_learning_bonus2_training_curve.png
│   ├── q_learning_bonus3_adaptive_gradient_curve.png
│   └── q_learning_bonus3_adaptive_tderror_curve.png
└── data/                  # 100 seed 原始评估数据 (JSON)
    ├── q_learning_report.json
    ├── reinforce_report.json
    ├── mcts_report.json
    ├── q_learning_bonus2_report.json
    ├── smooth_v1_eval_0.00.json      # Smooth 验证数据
    ├── smooth_v1_eval_0.05.json
    ├── adaptive_gradient_eval_0.00.json
    ├── adaptive_gradient_eval_0.05.json
    ├── adaptive_tderror_eval_0.00.json
    ├── adaptive_tderror_eval_0.05.json
    └── bonus3_report.json
```

---

## 六、训练脚本

| 脚本 | 说明 |
|------|------|
| `train_qlearning.py` | 基础 Q-learning |
| `train_reinforce.py` | REINFORCE 策略梯度 |
| `train_mcts.py` | MCTS 树搜索 |
| `train_qlearning_bonus2.py` | 优化 Q-learning (12 bins, 衰减 ε) |
| `train_qlearning_bonus3_smooth.py` | 邻域平滑 (失败) |
| `train_qlearning_bonus3_hedge_v2.py` | Hedge 集成 (失败) |
| `train_qlearning_bonus3_adaptive_gradient.py` | **梯度驱动自适应 (成功)** |
| `train_qlearning_bonus3_adaptive_tderror.py` | TD 误差驱动自适应 (失败) |
| `evaluate.py` | 通用评估入口 (禁改) |
| `bonus3_analysis.py` | 鲁棒性分析绘图 (使用真实 JSON) |
| `verify_hedge.py` | Hedge 对比验证脚本 |
