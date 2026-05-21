# CartPole 强化学习课程项目报告

## 一、评分逻辑说明

**评估环境**: `gymnasium.CartPole-v1`
- 状态空间: 4维连续值 (cart位置、cart速度、pole角度、pole角速度)
- 动作空间: 2维离散 (左推/右推)
- **每存活1步，环境返回 reward = +1**
- **分数 = episode 内累计步数 = 累计回报**

**评估设置** (evaluate.py):
- `max_episode_steps = 2000` (覆盖环境默认的 500)
- 100个不同 seed (42~141)
- 初始状态加入高斯扰动 `perturb_scale` 测试鲁棒性
- 评估时 epsilon 强制设为 0 (greedy)

**关键指标**:
- `mean`: 100 seed 的平均步数
- `median`: 中位数
- `std`: 标准差
- `reached_max_ratio`: 达到 2000 步上限的比例

---

## 二、基础算法 (Bonus 1)

三种基础算法在 100 seed 下的评估结果 (perturb=0):

| 算法 | mean | median | std | max_ratio |
|------|------|--------|-----|-----------|
| **Q-learning** | 205.7 | 198 | 31.4 | 0% |
| **REINFORCE** | **2000.0** | **2000** | **0.0** | **100%** |
| **MCTS** | 233.1 | 231 | 11.6 | 0% |

**结论**: REINFORCE 以策略梯度直接拟合连续状态，完美达到上限。Q-learning 受限于 6×6×6×6=1296 状态离散化，表达能力不足。MCTS 因搜索深度和模拟次数受限，表现一般。

---

## 三、Bonus 2: Q-learning 优化

在基础 Q-learning 上做了以下改进:
- bin 数从 6 提升到 **12** (12⁴=20736 状态)
- epsilon 从固定 0.1 改为 **衰减** (1.0 → 0.01)
- 奖励从稀疏 (仅结束给) 改为 **稠密** (每步 +1，失败惩罚)
- 训练轮数提升到 **15000**

| 指标 | 数值 |
|------|------|
| mean | **1790.6** |
| median | **2000** |
| std | 504.6 |
| max_ratio | **81%** |

---

## 四、Bonus 3: 鲁棒性改进

在 Q-learning (6 bins, 2000 epochs) 基础上，测试了 5 种方法在 perturb=0 和 perturb=0.05 下的表现。

### 4.1 结果总览

| 方法 | perturb=0 | perturb=0.05 | vs 基线提升 |
|------|-----------|--------------|------------|
| **Baseline** (固定均匀离散化) | 205.7 | 194.1 | — |
| **Smooth** (邻域线性插值) | 461.8 | 306.3 | +57.8% |
| **Hedge** (多表加权集成) | 330.1 | 331.1 | +70.6% |
| **AdaptiveGrad** (梯度驱动自适应边界) | **1906.0** | **1865.8** | **+861.3%** |
| **AdaptiveTD** (TD误差驱动自适应边界) | 145.6 | 127.2 | -34.5% |

**30% hard threshold** (perturb=0.05 需 ≥ 252.3):
- Smooth: PASS (+57.8%)
- Hedge: PASS (+70.6%)
- **AdaptiveGrad: PASS (+861.3%)**
- AdaptiveTD: FAIL (-34.5%)

### 4.2 各方法说明与关键代码

#### 4.2.1 邻域平滑 (Smooth)

**思想**: 决策时不只用当前 bin 的 Q 值，而是对周围 16 个相邻 bin 做三线性插值，使策略表面更平滑，减少边界抖动。

```python
def _smooth_q_values(self, current_state):
    fractions = self._get_bin_fractions(current_state)
    q_vals = np.zeros(self.n_action)
    for offset_bits in range(16):  # 2^4 个邻居
        bin_indices = []
        w = 1.0
        for dim in range(4):
            bin_idx, frac = fractions[dim]
            if (offset_bits >> dim) & 1:
                neighbor_idx = min(bin_idx + 1, MAX_DIGITIZE_NUM - 1)
                w *= frac  # 线性权重
            else:
                neighbor_idx = bin_idx
                w *= (1.0 - frac)
            bin_indices.append(neighbor_idx)
        state_id = ...
        q_vals += w * self.q_table[state_id, :]
    return q_vals
```

**结果**: perturb=0.05 时 306.3 步，比基线提升 57.8%。

---

#### 4.2.2 Hedge 多表集成 (Hedge)

**思想**: 训练 4 张独立 Q 表，每张表的 bin 边界有不同偏移 (0, 0.25, 0.5, 0.75 bin 宽度)。用 Hedge 算法在线调整各表权重。

**Hedge 更新**:
```python
def update_hedge(self, state, action_taken, ...):
    losses = np.zeros(NUM_TABLES)
    for i, table in enumerate(self.tables):
        probs = table.get_probs(state, temperature=1.0)
        losses[i] = 1.0 - probs[action_taken]  # 不支持该动作则 loss 高
    self.hedge_weights *= np.exp(-self.hedge_eta * losses)
    self.hedge_weights /= np.sum(self.hedge_weights)
```

**结果**: perturb=0.05 时 331.1 步，提升 70.6%。但权重最终坍缩到单表，集成效果有限。

---

#### 4.2.3 梯度驱动自适应离散化 (AdaptiveGrad) — **最佳方法**

**思想**: 同样的 6⁴=1296 个状态格子，但 bin **边界位置**根据 Q 值梯度动态调整。Q 值变化剧烈的区域 bin 更密，变化平缓的区域 bin 更疏。

**核心逻辑 — 边界重分配**:
```python
def _adapt_boundaries(self):
    for dim in range(4):
        # 1. 计算每维各 bin 的平均 max-Q
        q_per_bin = [mean(max_Q(bin_states)) for b in range(6)]
        
        # 2. 相邻 bin 的梯度 = |ΔQ|
        gradients = [abs(q_per_bin[b+1] - q_per_bin[b]) + 0.005 
                     for b in range(5)]
        
        # 3. 密度 = 1 + alpha * gradient，累积后均匀重采样边界
        #    高梯度区 -> 密度高 -> bin 更密
        cum_density = [0.0]
        for b in range(6):
            width = boundaries[b+1] - boundaries[b]
            dens = 1.0 + GRAD_ALPHA * gradients[min(b, len(gradients)-1)]
            cum_density.append(cum_density[-1] + dens * width)
        
        # 在累积密度的均匀分位点放置新边界
        new_boundaries = [lo]
        for i in range(1, 6):
            target = cum_density[-1] * i / 6
            # 线性插值找位置
            ...
        new_boundaries.append(hi)
```

**Q 表迁移** (边界改变后保留学习成果):
```python
def _remap_q_table(self, old_boundaries, momentum=0.3):
    old_q = self.q_table.copy()
    new_q = np.zeros_like(self.q_table)
    for state_id in range(MAX_DIGITIZE_NUM ** 4):
        center = self._get_bin_center(state_id, self.bin_boundaries)
        old_id = self._lookup_with_boundaries(center, old_boundaries)
        new_q[state_id] = old_q[old_id]
    self.q_table = momentum * self.q_table + (1.0 - momentum) * new_q
```

**为什么有效**:
- 固定 1296 个格子，但将分辨率集中到最需要的地方
- CartPole 的平衡点附近 Q 值变化剧烈，AdaptiveGrad 自动在该区域加密
- 每 250 episode 调整一次，共调整 7 次，渐进优化

**结果**: perturb=0 时 mean=1906.0 (82%达上限)，perturb=0.05 时 mean=1865.8 (79%达上限)。

---

#### 4.2.4 TD 误差驱动自适应离散化 (AdaptiveTD)

**思想**: 与 AdaptiveGrad 类似，但用累积 TD 误差代替 Q 值梯度作为密度函数。

```python
# 更新时记录 TD 误差
self.td_errors[current_id] = 0.9 * self.td_errors[current_id] + 0.1 * abs(td_error)

# 调整时密度 = 1 + TD_ALPHA * td_error_per_bin
```

**失败原因**:
- 训练初期 TD 误差普遍很大，边界被吸引到不稳定区域
- 高 TD 误差的区域不一定是"需要更高分辨率"的区域，可能只是噪声
- 频繁的边界调整+Q表迁移导致学习过程被反复打断

**结果**: perturb=0.05 时仅 127.2 步，低于基线。

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
│   ├── q_learning_bonus3_smooth_training_curve.png
│   ├── q_learning_bonus3_adaptive_gradient_curve.png
│   └── q_learning_bonus3_adaptive_tderror_curve.png
└── data/                  # 100 seed 原始评估数据 (JSON)
    ├── q_learning_report.json
    ├── reinforce_report.json
    ├── mcts_report.json
    ├── q_learning_bonus2_report.json
    ├── adaptive_gradient_eval_0.00.json
    ├── adaptive_gradient_eval_0.05.json
    ├── adaptive_tderror_eval_0.00.json
    ├── adaptive_tderror_eval_0.05.json
    └── bonus3_report.json
```

---

## 六、训练脚本

完整训练脚本位于 `cartpole_course_project/` 目录下:
- `train_qlearning.py` — 基础 Q-learning
- `train_reinforce.py` — REINFORCE 策略梯度
- `train_mcts.py` — MCTS 树搜索
- `train_qlearning_bonus2.py` — 优化 Q-learning
- `train_qlearning_bonus3_smooth.py` — 邻域平滑
- `train_qlearning_bonus3_hedge.py` — Hedge 集成
- `train_qlearning_bonus3_adaptive_gradient.py` — 梯度驱动自适应离散化 (最佳)
- `train_qlearning_bonus3_adaptive_tderror.py` — TD 误差驱动自适应离散化
- `evaluate.py` — 通用评估入口
- `bonus3_analysis.py` — 鲁棒性分析绘图
