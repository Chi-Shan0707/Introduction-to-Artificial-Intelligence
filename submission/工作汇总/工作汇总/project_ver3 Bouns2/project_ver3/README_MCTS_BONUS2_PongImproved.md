# MCTS Bonus 2：CartPole 在线规划强化优化版

本项目实现了一个面向 `CartPole-v1` 的 Monte Carlo Tree Search（MCTS）在线规划智能体。与 Q-Learning 或 REINFORCE 这类“先训练、再评测”的方法不同，MCTS 在每一步决策时从当前状态出发即时搜索未来轨迹，并根据模拟结果选择当前动作。

该版本面向 Bonus 2 性能优化目标，对基础 MCTS 进行了系统改造：将“随机 rollout + 大尺度 `C_p`”优化为“内部动力学模型 + 归一化价值 + 控制器引导 rollout + PUCT 先验 + 根节点复核”的组合方案。

## 1. 项目背景

CartPole 的目标是通过左右推动小车，让杆子尽可能长时间保持直立。状态包含 4 个连续变量：

| 状态变量 | 含义 |
|---|---|
| `cart_pos` | 小车位置 |
| `cart_vel` | 小车速度 |
| `pole_angle` | 杆子角度 |
| `pole_angular_velocity` | 杆子角速度 |

动作空间为：

```text
0 = 向左推
1 = 向右推
```

本实现使用：

```python
ENV_ID = "CartPole-v1"
ENV_MAX_STEPS = 2000
```

单局最高分为 2000。

## 2. 文件与接口

建议将代码文件命名为：

```text
train_mcts_bonus2_PongImproved.py
```

或更规范地重命名为：

```text
train_mcts_bonus2_optimized.py
```

如果使用 `evaluate.py` 动态加载，文件名必须和 `--agent-class` 中的模块名一致。

MCTS 不需要 checkpoint。代码中的 `load_model()` 是为了兼容统一评测接口而保留的空桩方法。

## 3. 核心实现

### 3.1 搜索节点 MCTSNode

每个搜索树节点包含：

```python
self.params   # 当前连续状态
self.children # 子节点字典
self.parent   # 父节点
self.Q        # 累计价值
self.N        # 访问次数
self.done     # 是否终止
self.depth    # 当前深度
self.action   # 从父节点到当前节点的动作
```

其中 `params` 直接保存 CartPole 的连续状态，因此 MCTS 不需要像 Q-Learning 那样先离散化状态。

### 3.2 每步决策流程

每次 `predict(state)` 被调用时，智能体执行一次 UCT 搜索：

1. 根据当前状态建立或复用根节点；
2. 确保根节点的两个动作都至少扩展一次；
3. 在给定 `iteration_budget` 内重复执行：
   - Selection：选择最有潜力的子节点；
   - Expansion：扩展新动作；
   - Rollout：用引导策略模拟未来；
   - Backpropagation：把价值回传到路径上的节点；
4. 用根节点复核机制选择最终动作；
5. 将被选中的子节点作为下一步缓存节点，实现跨 step 树复用。

## 4. 关键优化思路

### 优化一：内部动力学模型替代 Gym rollout

基础 MCTS 如果每次 rollout 都调用 `gym.step()` 或新建环境，速度会很慢。MCTS 的性能高度依赖单位时间内能模拟多少未来轨迹，因此模拟效率非常关键。

本版本手写了 CartPole 的一步动力学：

```python
_model_step(state, action, depth)
```

该函数根据 CartPole 物理参数计算下一状态，包括：

- 重力；
- 小车质量；
- 杆子质量；
- 杆长；
- 推力大小；
- Euler 积分；
- 小车位置终止边界；
- 杆子角度终止边界。

这使得树搜索和 rollout 可以直接在内部模型中高速完成，不需要频繁调用真实环境。

### 优化二：归一化 rollout value，修正 UCT 尺度问题

基础 MCTS 中，如果 rollout 返回原始存活步数，那么 `Q/N` 的尺度可能是几十、几百甚至上千；而 UCT 探索项的尺度取决于 `C_p`。这会导致两个问题：

- `C_p` 太小：搜索过早贪心；
- `C_p` 太大：搜索过度探索，忽视已有高价值分支。

本版本将 value 归一化：

```python
value = depth / ENV_MAX_STEPS
```

再加入稳定性奖励和失败惩罚：

```python
value += STABILITY_BONUS_WEIGHT * stability_score
value -= FAILURE_PENALTY * early_failure_penalty
```

这样 `Q/N` 大致落在 `[0, 1]` 附近，`C_p` 可以设置为更合理的小尺度：

```python
C_P_INIT = 1.25
C_P_MIN = 0.15
C_P_MAX = 3.00
```

这是该版本相对于基础 MCTS 的核心改进之一。

### 优化三：控制器引导 rollout

随机 rollout 对 CartPole 来说质量较低，因为随机左右推通常很快导致失败，无法提供足够有区分度的未来价值估计。

本版本定义了一组线性控制器：

```python
self.policy_weights = [
    [-0.35, -0.65, 7.5, 1.10],
    [-0.50, -0.85, 9.0, 1.35],
    [-0.70, -1.05, 11.0, 1.65],
    [-0.95, -1.35, 13.0, 2.00],
    [-1.20, -1.60, 15.0, 2.35],
    [0.00, 0.00, 1.0, 0.50],
]
```

动作选择为：

```python
action = 1 if dot(weights, state) > 0 else 0
```

这些控制器分别强调小车位置、小车速度、杆子角度和杆子角速度。它们不是直接替代 MCTS，而是让 rollout 更像“有基本控制能力的模拟”，从而提供更有效的估值。

### 优化四：PUCT 风格先验引导

在 `_bestchild()` 中，本版本对普通 UCT 做了改造：

```python
prior = self._policy_prior(node.params, child.action)
exploration = C_p * prior * sqrt(parent_N) / (1 + child.N)
score = mean_value + exploration
```

如果某个动作与线性控制器推荐方向一致，它获得更高先验；否则先验较低。

这样做可以让有限搜索预算优先分配给更可能有价值的动作，而不是平均浪费在明显危险的分支上。

### 优化五：根节点最终动作复核

基础 MCTS 常用访问次数最多的 child 作为最终动作。但在有限预算下，访问次数多并不一定代表动作最好，也可能只是早期探索偏差导致。

因此本版本使用综合评分：

```python
TREE_VALUE_WEIGHT * tree_value
+ ROOT_VERIFY_WEIGHT * verify_value
+ 0.02 * prior
```

其中：

- `tree_value` 是树内平均价值；
- `verify_value` 是根节点短程安全复核价值；
- `prior` 是控制器先验。

权重为：

```python
TREE_VALUE_WEIGHT = 0.45
ROOT_VERIFY_WEIGHT = 0.55
```

这意味着最终动作不仅看树搜索统计，也看额外的短程安全验证。

### 优化六：Beam-MPC 根节点复核

普通 rollout 可能难以发现“小车慢慢漂到边界”的长期风险。为此代码加入短视野 beam search：

```python
USE_BEAM_MPC_ROOT = True
MPC_HORIZON = 18
MPC_BEAM_WIDTH = 64
```

Beam-MPC 会保留若干条局部最稳定的动作序列，用稳定性积分进行排序，并在最终动作选择时作为 tie-break 或 rescue。

它不替代 MCTS 的 selection、expansion、rollout、backup，而是专门补足根节点动作选择阶段的安全判断。

### 优化七：稳定性评分

代码实现了：

```python
_stability_score(state)
```

该函数根据以下因素给状态打分：

| 因素 | 含义 |
|---|---|
| `abs(x)` | 小车越远离中心越危险 |
| `abs(theta)` | 杆子角度越大越危险 |
| `abs(x_dot)` | 小车速度越大越可能冲向边界 |
| `abs(theta_dot)` | 杆子角速度越大越容易快速倒下 |

这让 rollout 不只区分“是否失败”，还能区分“虽然还没失败但已经很危险”的状态。

### 优化八：自适应 `C_p`

搜索结束后，代码根据本轮最大搜索深度调整探索系数：

```python
if max_depth < lookahead_target:
    C_p *= 0.97
else:
    C_p *= 1.015
C_p = clip(C_p, C_P_MIN, C_P_MAX)
```

含义是：

- 如果搜索不够深，减少探索，鼓励更深入地验证已有好分支；
- 如果搜索已经足够深，略微增加探索，保持多样性；
- 用上下界防止 `C_p` 失控。

相比基础版本的线性 `C_p += 1` / `C_p -= 1`，乘法调节更适合归一化 value。

### 优化九：跨 step 复用搜索树

`predict()` 中会检查外部状态是否与缓存的当前节点一致：

```python
mismatch = norm(state - self._current_node.params) > 1e-6
```

如果一致，就复用上一轮选中动作对应的子树；如果不一致或 episode 结束，则丢弃旧树。

这样可以避免每一步都从零开始搜索，提高在线规划效率。

## 5. 参数设置

| 参数 | 值 | 作用 |
|---|---:|---|
| `ITERATION_BUDGET` | 96 | 每步搜索迭代次数 |
| `C_P_INIT` | 1.25 | 初始探索系数 |
| `C_P_MIN` | 0.15 | 最小探索系数 |
| `C_P_MAX` | 3.00 | 最大探索系数 |
| `LOOKAHEAD_TARGET` | 260 | 搜索深度调节目标 |
| `ROLLOUT_DEPTH_LIMIT` | 420 | rollout 最大深度 |
| `ROOT_VERIFY_DEPTH` | 520 | 根节点复核深度 |
| `MPC_HORIZON` | 18 | Beam-MPC 视野 |
| `MPC_BEAM_WIDTH` | 64 | Beam-MPC 保留分支数 |
| `ENV_MAX_STEPS` | 2000 | 单局最大步数 |

## 6. 运行方式

### 直接运行

```bash
python train_mcts_bonus2_PongImproved.py
```

### 使用统一评测接口

```bash
python evaluate.py \
    --agent-class train_mcts_bonus2_PongImproved:Agent \
    --agent-init-kwargs '{"iteration_budget":96,"env_id":"CartPole-v1"}' \
    --seed-base 42 \
    --seed-count 100 \
    --max-episode-steps 2000
```

如果重命名为 `train_mcts_bonus2_optimized.py`，则使用：

```bash
python evaluate.py \
    --agent-class train_mcts_bonus2_optimized:Agent \
    --agent-init-kwargs '{"iteration_budget":96,"env_id":"CartPole-v1"}' \
    --seed-base 42 \
    --seed-count 100 \
    --max-episode-steps 2000
```

### 可视化

```bash
python vis.py \
    --agent-class train_mcts_bonus2_PongImproved:Agent \
    --agent-init-kwargs '{"iteration_budget":96,"env_id":"CartPole-v1"}' \
    --seed 42 \
    --output checkpoints/mcts_bonus2_seed42.gif
```

## 7. 与基础 MCTS 的对比

| 模块 | 基础 MCTS | 本优化版 | 优化目的 |
|---|---|---|---|
| rollout | 随机动作 | 多控制器引导 | 提高估值质量 |
| 环境模拟 | 依赖 Gym | 内部动力学模型 | 提高搜索速度 |
| value | 原始 reward / 步数 | 归一化 value | 修正 UCT 尺度 |
| `C_p` | 大尺度或线性调整 | 小尺度自适应 | 稳定探索项 |
| child 选择 | 普通 UCT | PUCT prior | 减少无效探索 |
| 最终动作 | 访问次数或均值 | 树价值 + 复核价值 | 降低早期偏差 |
| 漂移处理 | 较弱 | Beam-MPC 复核 | 防止小车慢慢偏离 |
| 搜索树 | 每步重建为主 | 跨 step 复用 | 提升在线效率 |

## 8. 设计取舍

### 为什么不直接使用手写控制器？

代码中的线性控制器主要用于 rollout、先验和复核。最终动作仍然由 MCTS 搜索树统计和根节点验证共同决定。因此它不是简单的规则控制器，而是“启发式引导的在线规划”。

### 为什么不单纯增加 iteration budget？

增加搜索次数确实可能提升效果，但会显著拖慢评测。该版本的优化重点不是暴力增大预算，而是提高每一次模拟的有效信息量：

- 内部模型让模拟更快；
- 控制器 rollout 让模拟更有意义；
- 归一化 value 让 UCT 更稳定；
- 根节点复核减少最终动作误选。

### 为什么 MCTS 路线更难？

CartPole 需要长期稳定控制，而 MCTS 是短视野在线规划。许多失败并非马上发生，而是当前动作导致小车缓慢漂移，几百步后才不可挽回。因此 MCTS 的难点在于如何让短期模拟看见长期风险。

本版本通过稳定性评分、Beam-MPC 和根节点复核来缓解这个问题。

## 9. 总结

该 MCTS Bonus 2 版本的核心思想是：

```text
不是简单增加搜索次数，而是提高每一次搜索模拟的质量。
```

具体优化包括：

1. 用内部动力学模型提升模拟速度；
2. 用归一化 value 修正 UCT 尺度失衡；
3. 用控制器引导 rollout 提高未来估计质量；
4. 用 PUCT 先验减少无意义探索；
5. 用根节点复核和 Beam-MPC 处理最终动作偏差与长期漂移；
6. 用自适应 `C_p` 平衡探索深度和搜索多样性。

整体上，这一版本展示了 MCTS 从“随机模拟搜索”向“模型驱动、启发式引导、稳定性约束在线规划”的优化过程。
