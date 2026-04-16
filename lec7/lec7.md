# 第7讲：深度学习（一）—— 神经网络基础知识

> 基于 `人工智能-7-深度学习（一）.pdf` 整理，丁恒辉，2026/4/16

---

## 0. 本讲主线

本讲是深度学习的入门课，三条主线：

1. **神经网络**：从线性分类器到多层非线性网络
2. **反向传播**：用计算图 + 链式法则高效求梯度
3. **正则化与优化**：防止过拟合、找到更好的解

---

## 1. 神经网络：从线性分类器到多层网络

### 1.1 出发点：线性分类器

线性分类器（如 SVM、Softmax）的得分函数：

$$s = Wx$$

$$x \in \mathbb{R}^d \to s \in \mathbb{R}^K$$

问题：线性分类器只能画超平面，无法处理非线性可分的数据（如红色点和蓝色点呈环形分布）。

### 1.2 两层神经网络

在输入和输出之间插入一个**隐藏层**，加上**非线性激活函数** $\sigma$：

$$s = W_2 \sigma(W_1 x)$$

$$x \in \mathbb{R}^{3072} \xrightarrow{W_1} h \in \mathbb{R}^{100} \xrightarrow{\sigma} \xrightarrow{W_2} s \in \mathbb{R}^{10}$$

**关键问题**：如果不加激活函数会怎样？

不加 $\sigma$ 的话：

$$s = W_2 (W_1 x) = (W_2 W_1) x = W'x$$

还是线性分类器！多层线性变换等价于单层线性变换。

**结论：非线性激活函数是让网络拥有强大表示能力的关键。**

### 1.3 激活函数

| 激活函数 | 公式 | 特点 |
|---|---|---|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | 输出 (0,1)，易梯度消失 |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 输出 (-1,1)，零中心 |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ | 计算快，缓解梯度消失 |
| Leaky ReLU | $\max(\alpha x, x), \alpha \ll 1$ | 解决 ReLU "死神经元"问题 |
| ELU | 类似 Leaky ReLU 但更平滑 | 输出均值更接近 0 |
| Maxout | $\max(w_1^T x + b_1, w_2^T x + b_2)$ | 学习激活函数本身 |

**实践**：在绝大多数任务上 ReLU 表现最好。

### 1.4 网络架构术语

- **2-layer Neural Net** = 1 个隐藏层（输入层不算，输出层算一层）
- **3-layer Neural Net** = 2 个隐藏层
- 通用术语：**全连接层（Fully-connected layer）**或**多层感知机（MLP）**

FC 层常用于网络的最后几层，完成分类、回归等具体任务。在图像等高维数据处理中，卷积层常取代 FC 层以减少参数；但在小规模任务和结构化数据中，FC 层仍是常用基础组件。

### 1.5 生物启发

神经网络的设计灵感来自生物神经元：

- **树突（Dendrite）**：接收信号 → 对应输入 $x$
- **胞体（Cell body）**：整合信号 → 对应求和 $w^T x + b$
- **轴突（Axon）**：传递信号 → 对应输出
- **突触（Synapse）**：连接权重 → 对应权重 $w$
- Sigmoid 激活函数本身就是对生物神经元"全或无"发放规律的建模

区别：生物神经元连接复杂且随机，而人工神经元为计算效率组织成规则的层次结构。不过研究表明，随机连接的神经网络也可以工作。

### 1.6 损失函数

将神经网络与损失函数结合：

$$L = \underbrace{L_{\text{data}}(W)}_{\text{数据损失}} + \lambda \underbrace{R(W)}_{\text{正则化}}$$

- 数据损失：模型预测应与训练数据匹配（SVM Loss / Softmax Loss）
- 正则化：防止过拟合
- 目标：求出最优的 $W_1, W_2$

---

## 2. 反向传播方法

### 2.1 为什么需要反向传播

手算梯度有两个致命问题：
1. **计算量过于庞大**
2. **有些函数无解析解**

因此需要系统化的方法：**计算图 + 链式法则**。

### 2.2 计算图

将任意复杂函数分解为基本运算的组合，每个运算是一个**节点**，数据流向构成**图**。

以 $f(x, y, z) = (x + y) \cdot z$ 为例：

```
x ──→ [+] ──→ q ──→ [×] ──→ f
y ──┘            z ──┘
```

- 前向传播（Forward pass）：从左到右计算每个节点的值
- 反向传播（Backward pass）：从右到左计算每个节点的梯度

### 2.3 链式法则

反向传播的核心数学工具：

$$\frac{\partial f}{\partial x} = \underbrace{\frac{\partial f}{\partial q}}_{\text{upstream gradient}} \times \underbrace{\frac{\partial q}{\partial x}}_{\text{local gradient}}$$

每个节点只需要知道两件事：
1. **上游梯度**（upstream gradient）：从后面传过来的梯度
2. **局部梯度**（local gradient）：该节点自身运算的导数

两者相乘 = **下游梯度**（downstream gradient），传给前面的节点。

### 2.4 例题：$f = (x+y)z$，其中 $x=-2, y=5, z=-4$

**前向传播**：
- $q = x + y = -2 + 5 = 3$
- $f = q \cdot z = 3 \times (-4) = -12$

**反向传播**：
- $\frac{\partial f}{\partial f} = 1$ （base case）
- $\frac{\partial f}{\partial z} = q = 3$ （乘法门：梯度互换）
- $\frac{\partial f}{\partial q} = z = -4$ （乘法门：梯度互换）
- $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x} = -4 \times 1 = -4$ （加法门：梯度直通）
- $\frac{\partial f}{\partial y} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial y} = -4 \times 1 = -4$ （加法门：梯度直通）

### 2.5 门电路的梯度规则

| 门 | 前向 | 反向规则 | 直觉 |
|---|---|---|---|
| **加法门** $f = x + y$ | $f = x + y$ | 梯度分配器：$\frac{\partial f}{\partial x}=1, \frac{\partial f}{\partial y}=1$ | 上游梯度直接复制给每个输入 |
| **乘法门** $f = x \cdot y$ | $f = x \cdot y$ | 梯度交换器：$\frac{\partial f}{\partial x}=y, \frac{\partial f}{\partial y}=x$ | 上游梯度乘以另一个输入的值 |
| **复制门** | 同一个值用多次 | 梯度累加器：各路径的梯度相加 | 反向传播时梯度会叠加 |
| **Max门** $f = \max(x, y)$ | 取较大值 | 梯度路由器：梯度只传给较大的那个 | 另一个输入对输出没有贡献 |

### 2.6 Sigmoid 的简化计算

Sigmoid 函数：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

展开为计算图需要很多节点，但可以**直接用局部梯度公式**简化：

$$(1 - \sigma(x)) \cdot \sigma(x)$$

例如 $\sigma(x) = 0.73$，则局部梯度为 $(1 - 0.73) \times 0.73 = 0.1971$。

### 2.7 完整反向传播实现（模块化）

```python
# 每个模块实现 forward() 和 backward() 接口

class MultiplyGate:
    def forward(self, x, y):
        self.x, self.y = x, y   # 缓存，供反向传播使用
        return x * y

    def backward(self, upstream_grad):
        # 上游梯度与局部梯度相乘
        return upstream_grad * self.y, upstream_grad * self.x

class AddGate:
    def forward(self, x, y):
        self.x, self.y = x, y
        return x + y

    def backward(self, upstream_grad):
        return upstream_grad, upstream_grad  # 梯度直通
```

**PyTorch 中的对应关系**：
- PyTorch 的每个运算符（operator）都自动实现了 `forward()` 和 `backward()`
- `autograd` 自动构建计算图并执行反向传播
- `forward`：计算结果，并缓存中间变量
- `backward`：应用链式法则计算梯度

### 2.8 反向传播总结

- 神经网络是线性函数和非线性激活函数的堆叠，比线性分类器具有更强的表示能力
- 反向传播：沿计算图递归应用链式法则，计算所有参数的梯度
- 实现上以模块化形式呈现，每个节点实现 forward / backward 接口

---

## 3. 正则化与优化

### 3.1 正则化

#### 3.1.1 为什么需要正则化

- 模型在训练数据上表现好，不代表在测试数据上也表现好（过拟合）
- 正则化通过对模型复杂度进行约束，平衡训练误差和模型复杂度

**奥卡姆剃刀**（Occam's Razor）：Among multiple competing hypotheses, the simplest is the best.

$$L = L_{\text{data}} + \lambda R(W)$$

$\lambda$ 是正则化强度（超参数），控制模型对训练数据的拟合程度。

#### 3.1.2 正则化方法分类

**简单方法**：

| 方法 | 惩罚项 | 效果 |
|---|---|---|
| L1 | $R(W) = \sum \|w\|$ | 部分参数变为零，实现特征选择 |
| L2 | $R(W) = \sum w^2$ | 参数趋近零但非零，限制模型复杂度 |
| Elastic Net | $R(W) = \alpha \sum \|w\| + (1-\alpha) \sum w^2$ | L1 + L2 的折中 |

**L1 vs L2 的权重偏好**：
- L2 倾向于**分散**的权重：$[0.25, 0.25, 0.25, 0.25]$ 比 $[1, 0, 0, 0]$ 更受青睐（因为 L2 的平方惩罚使得大值代价很高）
- L1 倾向于**稀疏**的权重：$[1, 0, 0, 0]$ 与 $[0.25, 0.25, 0.25, 0.25]$ 的 L1 惩罚相同（$|1|+0+0+0 = 0.25\times4$），但稀疏解更简单

**复杂方法**：

| 方法 | 核心机制 |
|---|---|
| Dropout | 训练时随机丢弃神经元 |
| Batch Normalization | 利用 mini-batch 统计量的随机噪声 |
| Stochastic Depth | 随机丢弃整条残差分支 |
| Fractional Pooling | 随机化池化操作 |

#### 3.1.3 为什么要使用正则化（三个理由）

1. **人为选择权重偏好**：让模型倾向于更简单、更合理的参数分布
2. **使模型简化**：减少对训练数据中噪声的拟合，提高泛化能力
3. **改善优化**：通过添加曲率（curvature）使损失面更平滑，帮助梯度下降找到更好的解

#### 3.1.4 L1/L2 损失函数的局限性

**L1（MAE）的局限**：
- 导数为常数（±1），训练后期在最小值附近波动，难以精确收敛
- $x=0$ 处不可导，不利于梯度优化

**L2（MSE）的局限**：
- 远离最优点时梯度很大，可能导致梯度爆炸

---

### 3.2 优化方法

#### 3.2.1 梯度计算

**数值梯度**（Numerical Gradient）：

$$\frac{\partial L}{\partial w_i} \approx \frac{L(w_i + h) - L(w_i - h)}{2h}$$

- 优点：容易编写
- 缺点：近似的、慢的（需要遍历所有维度）

**解析梯度**（Analytic Gradient）：

通过链式法则直接求导。

- 优点：精确的、快速的
- 缺点：容易出错

**实践**：始终使用解析梯度，但用数值梯度**检查**实现是否正确（称为梯度检查 Gradient Check）。

#### 3.2.2 梯度下降的三种变体

| 方法 | 每次使用的数据 | 特点 |
|---|---|---|
| Batch GD | 全部数据 | 精确但慢，一次迭代时间太长 |
| SGD | 单个样本 | 快但噪声大 |
| Mini-batch GD | 一个 batch（常用 32/64/128） | BGD 和 SGD 的折中，实践中最常用 |

#### 3.2.3 SGD 的问题

1. **病态条件数（Poor Conditioning）**：损失函数在不同方向上变化速度不同，Hessian 矩阵的最大奇异值与最小奇异值的比值很大。导致在浅方向（shallow）进展缓慢，在陡方向（steep）波动剧烈。
2. **局部最小值和鞍点**：梯度为零时，SGD 会停滞。高维空间中鞍点比局部最小值更常见。
3. **梯度噪声**：mini-batch 的梯度只是真实梯度的估计值，包含采样噪声。

#### 3.2.4 SGD + Momentum

核心思想：模拟物理中的"惯性"，建立速度变量来平滑梯度更新。

$$v_t = \rho \cdot v_{t-1} + \nabla L$$
$$W_{t+1} = W_t - \alpha \cdot v_t$$

- $\rho$ 控制速度平滑程度，通常设为 0.9 或 0.99
- 效果：加速收敛、减少震荡、帮助逃出局部最小值和鞍点

```python
# Python 实现
v = 0
rho = 0.9
while True:
    dw = compute_gradient(W)
    v = rho * v + dw
    W -= learning_rate * v
```

#### 3.2.5 RMSProp

核心思想：自适应学习率，根据梯度的历史幅度动态调整每个参数的学习率。

$$v_t = \rho \cdot v_{t-1} + (1 - \rho) (\nabla L)^2$$
$$W_{t+1} = W_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \cdot \nabla L$$

- 梯度大的方向（steep）：$v_t$ 大，实际学习率被抑制
- 梯度小的方向（flat）：$v_t$ 小，实际学习率被加速
- 直觉上：在陡方向上踩刹车，在平坦方向上踩油门

#### 3.2.6 Adam

Adam = **Momentum + RMSProp**，目前最常用的深度学习优化器。

同时利用梯度的一阶矩（均值，即 Momentum）和二阶矩（方差，即 RMSProp）来自适应调整每个参数的学习率。

```python
# Adam 核心步骤
m = 0   # 一阶矩（梯度均值）
v = 0   # 二阶矩（梯度方差）
beta1, beta2, eps = 0.9, 0.999, 1e-8

while True:
    dw = compute_gradient(W)
    m = beta1 * m + (1 - beta1) * dw          # 更新一阶矩
    v = beta2 * v + (1 - beta2) * dw**2       # 更新二阶矩
    m_hat = m / (1 - beta1**t)                # 偏差校正
    v_hat = v / (1 - beta2**t)                # 偏差校正
    W -= learning_rate * m_hat / (sqrt(v_hat) + eps)
```

**注意**：AdamW 是 Adam 的改进版，将权重衰减（weight decay）从梯度更新中解耦出来。

#### 3.2.7 学习率调度

所有优化器都有学习率作为超参数，但没有"哪个最好"的统一答案，每种都有适用场景。常见的学习率调度策略：

| 策略 | 说明 |
|---|---|
| Step decay | 在固定轮次（如 30, 60, 90 epoch）将学习率乘以 0.1 |
| Cosine annealing | $\eta_t = \frac{\eta_0}{2}(1 + \cos(\frac{\pi t}{T}))$，余弦衰减 |
| Linear warmup | 前 ~5000 步将学习率从 0 线性增加到初始值，避免初始梯度爆炸 |
| Inverse sqrt | $\eta_t = \eta_0 / \sqrt{t}$，倒数平方根衰减 |

**经验法则**：批量大小增加 $N$ 倍时，初始学习率也应按 $N$ 的比例缩放。

---

## 4. 深度思考：Batch Normalization 为什么是正则化

### 4.1 BN 的原始设计目标

Batch Normalization 的设计初衷是**加速训练**，通过归一化每层的输入分布来缓解**内部协变量偏移（Internal Covariate Shift）**，让优化更稳定。

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

### 4.2 为什么它同时起到正则化作用

**原因一：Mini-batch 引入的随机噪声**

$\mu_B$ 和 $\sigma_B$ 是当前 mini-batch 的统计量。每次训练时，同一个样本所在的 mini-batch 不同，所以归一化结果每次都不同。这等价于给每个样本添加了随机扰动，模型无法死记硬背任何单个样本的精确值。

**原因二：类比 Dropout**

| | Dropout | Batch Normalization |
|---|---|---|
| 机制 | 随机丢弃神经元 | 随机扰动特征值 |
| 训练时 | 网络的"样子"每次不同 | 特征的"尺度"每次不同 |
| 测试时 | 关闭随机性 | 用 running mean/var 替代 mini-batch 统计量 |

两者都是**训练时注入随机性，测试时去随机性**。

**原因三：梯度噪声帮助找到平坦最优解**

mini-batch 统计量的随机性使梯度被注入噪声，迫使模型在更大的解空间区域内寻找解，倾向于找到**更平坦的最优解**（泛化能力更强）。

### 4.3 BN 的局限与 LayerNorm

BN 按 batch 维度归一化，适合 CV 中固定尺寸的图像数据。但在 NLP 中：
- 序列长度不固定
- batch 维度的统计量不稳定

因此 Transformer 使用 **Layer Normalization** 替代 Batch Normalization：

$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \cdot \gamma + \beta$$

LayerNorm 按单个样本的特征维度归一化，不依赖 mini-batch，因此随机性远小于 BN，正则化效果也弱得多。

---

## 5. 深度思考：Transformer 中的正则化

### 5.1 Transformer 使用的正则化方法

| 方法 | 使用位置 | 作用 |
|---|---|---|
| **Dropout** | Attention 权重后、FFN 中间层、Embedding 层、输出层前 | 最主要的正则化手段，默认 $p=0.1$ |
| **Layer Normalization** | 每个子层之后 | 训练稳定性 + 隐式正则化 |
| **Label Smoothing** | 训练标签 | 不让模型 100% 自信，提高泛化 |
| **Weight Decay (L2)** | AdamW 中 | 对权重施加 L2 惩罚 |
| **DropPath / Stochastic Depth** | 深层 Transformer（ViT、BERT） | 随机丢弃整条残差分支 |

**关键结论**：Transformer **没有使用 BatchNorm**，而是用 LayerNorm 替代。正则化主要由 Dropout 和 Weight Decay 承担。

---

## 6. 深度思考：Stochastic Depth 与 Fractional Pooling 详解

### 6.1 Stochastic Depth（随机深度）

#### 6.1.1 动机：深层网络的冗余问题

ResNet 引入残差连接后，网络可以堆叠到上百层甚至上千层。但一个关键观察是：

> **深层网络中并非所有层都是同等重要的。**

实验发现：
- 浅层提取低级特征（边缘、纹理），几乎所有样本都需要
- 深层提取高级语义（物体部件、复杂模式），但并非每个样本都需要全部深层
- 训练中期，梯度经过大量层的反向传播，前层获得的梯度信号非常微弱

问题本质：**网络越深，冗余越大，过拟合风险越高，训练也越困难。**

#### 6.1.2 核心思想

Stochastic Depth（Huang et al., 2016 "Deep Networks with Stochastic Depth"）的做法非常直接：

**训练时，以概率 $p_l$ 随机丢弃整条残差分支，只保留恒等映射（identity shortcut）。**

对于一个残差块：

$$\text{正常: } y = \mathcal{F}(x, W_l) + x$$

$$\text{Stochastic Depth: } y = \begin{cases} \mathcal{F}(x, W_l) + x & \text{以概率 } 1 - p_l \\ x & \text{以概率 } p_l \end{cases}$$

注意这里与 Dropout 的关键区别：
- **Dropout**：随机丢弃**单个神经元**（细粒度）
- **Stochastic Depth**：随机丢弃**整一层**的变换（粗粒度）

#### 6.1.3 丢弃概率的设置：线性衰减策略

最常用的策略是**线性衰减**：

$$p_l = 1 - \frac{L - l}{L(1 - p_L)}$$

其中：
- $L$ 是总层数
- $l$ 是当前层编号
- $p_L$ 是最深层的目标丢弃概率（如 0.5）

直观理解：

```
浅层 (l ≈ 1):   p_l ≈ 0   → 几乎不丢弃（浅层很重要，保留低级特征）
深层 (l ≈ L):   p_l ≈ 0.5 → 高概率丢弃（深层冗余多，需要正则化）
```

这符合直觉：浅层是"基础设施"，深层是"专家层"，基础设施不能随便关，但专家可以轮流上班。

#### 6.1.4 测试时的处理

测试时不能丢弃任何层（否则输出不确定）。使用**所有残差块的加权组合**：

$$y = \prod_{l'=l}^{L}(1 - p_{l'}) \cdot \mathcal{F}(x, W_l) + x$$

但在实践中，更常见的做法是**直接使用全部层**（类似 Dropout 测试时关闭），因为线性衰减使得丢弃概率通常不大，加权校正的影响较小。

#### 6.1.5 为什么 Stochastic Depth 有效

**效果一：正则化**

每次训练时，网络实际上是一个**随机的子网络**（类似于 Dropout 产生 $2^n$ 个子网络）。这防止了整个深层网络协同过拟合训练数据。

**效果二：隐式集成（Implicit Ensemble）**

训练过程中，不同的 mini-batch 使用不同的子网络。最终测试时用完整网络，等价于所有子网络的**隐式集成**——这类似于 Bagging 的思想，但不需要训练多个独立模型。

**效果三：减少训练时间**

深层有 50% 概率被跳过，意味着前向/反向传播的计算量直接减半。在达到相同精度的条件下，Stochastic Depth 的训练时间比完整 ResNet 少约 25-50%。

**效果四：缓解梯度消失**

因为浅层经常直接通过恒等映射连接到输出（中间层被跳过），梯度可以更直接地回传到浅层，缓解了深层网络的梯度消失问题。

#### 6.1.6 现代应用

Stochastic Depth 在现代大模型中被广泛使用，但名称和实现有所变化：

| 论文 | 名称 | 应用场景 |
|---|---|---|
| Huang et al., 2016 | Stochastic Depth | ResNet |
| Touvron et al., 2021 (DeiT) | DropPath | Vision Transformer (ViT) |
| Liu et al., 2021 (Swin) | DropPath | Swin Transformer |

在 ViT 中，DropPath 的典型设置是：
- 最前面的几个 block：$p = 0$
- 最后面的 block：$p = 0.1 \sim 0.5$（按层号线性增长）

#### 6.1.7 伪代码

```python
def stochastic_depth(x, block, p, training=True):
    """
    x: 输入特征
    block: 残差块 (函数或模块)
    p: 该层被丢弃的概率
    """
    if not training or p == 0.0:
        return block(x) + x

    # 以概率 p 跳过该 block，只保留恒等映射
    if random.random() < p:
        return x  # 直接返回输入，block 被完全跳过
    else:
        return block(x) + x

# 训练循环中使用线性衰减
for l, block in enumerate(blocks):
    p_l = linear_decay(l, total_layers, max_p=0.5)
    x = stochastic_depth(x, block, p_l, training=True)
```

---

### 6.2 池化（Pooling）基础

在讲解 Fractional Pooling 之前，先理解什么是池化。

#### 6.2.0 什么是池化

**池化是一种下采样操作**：在保持重要信息的同时，缩小特征图的空间尺寸（高度 × 宽度）。

直观类比：把一张高分辨率图片缩小成缩略图——你看不到每个像素了，但还能认出照片里是什么。池化做的就是这个事情。

为什么需要池化？三个原因：

1. **减少计算量**：特征图尺寸减半，后续层的参数量和计算量大幅下降
2. **扩大感受野**：池化后一个输出像素对应输入中更大的区域，让高层特征"看到"更大的范围
3. **提供平移不变性**：即使物体在图像中移动了几个像素，池化后的特征变化很小

#### 6.2.1 标准池化操作

**Max Pooling（最大池化）**：在每个窗口内取最大值。

$$y_{i,j} = \max_{(m,n) \in \text{window}(i,j)} x_{m,n}$$

```
输入 4×4:                    Max Pooling 2×2, stride 2:
┌──┬──┬──┬──┐
│1 │3 │2 │1 │                ┌───┬───┐
├──┼──┼──┼──┤                │ 6 │ 4 │
│5 │6 │3 │0 │       →        ├───┼───┤
├──┼──┼──┼──┤                │ 8 │ 5 │
│7 │2 │1 │8 │                └───┴───┘
├──┼──┼──┼──┤
│0 │1 │5 │2 │
└──┴──┴──┴──┘

左上窗口 max(1,3,5,6)=6   右上窗口 max(2,1,3,0)=4
左下窗口 max(7,2,0,1)=7→8  右下窗口 max(1,8,5,2)=5
```

直觉：保留窗口内**最显著**的特征（最亮的点、最强的边缘）。

**Average Pooling（平均池化）**：在每个窗口内取均值。

$$y_{i,j} = \frac{1}{|\text{window}|} \sum_{(m,n) \in \text{window}(i,j)} x_{m,n}$$

```
左上窗口 avg(1,3,5,6) = 3.75   右上窗口 avg(2,1,3,0) = 1.5
左下窗口 avg(7,2,0,1) = 2.5    右下窗口 avg(1,8,5,2) = 4.0
```

直觉：保留窗口内的**整体信息**，而不是只关注最突出的点。

**Global Average Pooling（全局平均池化）**：窗口覆盖整个特征图，每个通道只输出一个标量。

$$y_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j,c}$$

在现代网络（如 ResNet）中，GAP 常用来替代最后的全连接层：$H \times W \times C \to 1 \times 1 \times C$，大幅减少参数。

#### 6.2.2 Max Pooling vs Average Pooling

| | Max Pooling | Average Pooling |
|---|---|---|
| 选取策略 | 窗口内最大值 | 窗口内均值 |
| 保留什么 | 最显著的特征 | 整体统计信息 |
| 对噪声 | 鲁棒（只取最大值，噪声被忽略） | 敏感（噪声被平均进去） |
| 平移不变性 | 强（只要最大值还在窗口内就行） | 较弱（整体分布会变） |
| 梯度 | 只传给最大值所在位置（稀疏梯度） | 均匀分配给窗口内所有位置（密集梯度） |
| 典型用途 | 卷积网络中间层（提取关键特征） | 网络末端（GAP 替代 FC 层） |

#### 6.2.3 池化的两个超参数

- **窗口大小（kernel size）**：池化窗口的尺寸，如 $2 \times 2$、$3 \times 3$
- **步长（stride）**：窗口每次移动的距离

常见配置：
- kernel = $2 \times 2$, stride = 2 → 尺寸减半（最常用）
- kernel = $3 \times 3$, stride = 2 → 尺寸约为原来的 $1/2$
- kernel = $3 \times 3$, stride = 1 → 尺寸不变（较少见）

#### 6.2.4 现代趋势：用 stride 卷积替代池化

在 ResNet 等现代网络中，显式的池化层越来越少。取而代之的是用 **stride > 1 的卷积** 同时完成特征提取和下采样：

```
传统做法:  Conv(stride=1) → ReLU → Pool(stride=2)    ← 两步
现代做法:  Conv(stride=2) → ReLU                       ← 一步完成
```

优点：下采样的方式可以被学习（卷积核的参数是可训练的），而不是固定取最大值或均值。

但池化层并非完全消失——在特征图需要大幅缩小（如 $224 \times 224 \to 7 \times 7$）时，池化仍然是最简单高效的选择。

#### 6.2.5 背景：标准池化的局限（Fractional Pooling 的出发点）

标准池化操作是确定性的：

| 池化类型 | 做法 | 特点 |
|---|---|---|
| Max Pooling | 在每个区域内取最大值 | 确定性，固定下采样 |
| Average Pooling | 在每个区域内取均值 | 确定性，固定下采样 |

例如，$4 \times 4$ 的特征图用 $2 \times 2$ 的窗口做 Max Pooling，得到 $2 \times 2$ 的输出。输出中每个值的位置完全由输入决定，没有任何随机性。

**问题**：这种确定性可能导致两个后果：
1. 同样的输入永远产生同样的中间表示，模型容易对特定模式产生过度依赖
2. 池化窗口的位置是固定的（左上、右上、左下、右下），如果重要信息恰好落在窗口边界，就会被割裂

#### 6.2.2 核心思想

Fractional Pooling（Boureau et al., 2010）引入**随机性到池化过程**中，使下采样的位置和比例都带有随机性。

核心改变：**不再用固定大小的非重叠窗口，而是用随机采样的方式从输入中挑选像素。**

#### 6.2.3 两种实现方式

**方式一：Random Pooling（随机池化）**

与 Max Pooling 窗口相同，但窗口内的值不是取最大值，而是**按值的大小加权随机采样**：

```python
def random_pool(region):
    # region: 池化窗口内的值，如 [3, 1, 7, 2]
    weights = softmax(region)  # 按值大小转成概率，如 [0.002, 0.000, 0.998, 0.000]
    return random_sample(region, weights)  # 以 99.8% 概率选 7，但也可能选 3
```

直觉：值大的像素更可能被选中，但不总是选最大的。这既保留了 Max Pooling "选重要特征"的倾向，又引入了随机性。

**方式二：Fractional Max Pooling（分数最大池化）**

这是更核心的形式。标准 Max Pooling 只能做整数倍下采样（$2\times \to 1\times$, $3\times \to 1\times$），而 Fractional Max Pooling 允许**非整数倍下采样**（如 $7 \times 7 \to 3 \times 3$）。

做法：
1. 在输入的行和列上**随机生成切分点**（pseudorandom）
2. 每个切分区域内的值取最大值
3. 不同前向传播，切分位置不同，因此输出不同

```
标准 Max Pooling (7×7 → 3×3)：        Fractional Max Pooling (7×7 → 3×3)：

┌───┬───┬───┐                          ┌─────┬───┬───┐
│   │   │   │  ← 固定切分              │     │   │   │  ← 随机切分
├───┼───┼───┤                          │     ├───┼───┤
│   │   │   │                          │     │   │   │
├───┼───┼───┤                          ├─────┤   │   │
│   │   │   │                          │     │   │   │
└───┴───┴───┘                          └─────┴───┴───┘
每次都一样                              每次切分位置不同
```

#### 6.2.4 随机切分点的生成

切分点通过**伪随机**方式生成（保证可复现），同时满足约束：
- 切分点必须在合法范围内
- 每个区域的面积大致均匀（但允许小幅随机偏移）

```python
def fractional_max_pool(input, output_size):
    """
    input: H x W 的特征图
    output_size: (h_out, w_out)
    """
    H, W = input.shape

    # 在高度方向上随机生成 h_out 个区间
    # 每个区间长度 ≈ H / h_out，但有随机偏移
    row_intervals = pseudorandom_split(H, h_out)

    # 在宽度方向上随机生成 w_out 个区间
    col_intervals = pseudorandom_split(W, w_out)

    output = np.zeros(output_size)
    for i, (r_start, r_end) in enumerate(row_intervals):
        for j, (c_start, c_end) in enumerate(col_intervals):
            output[i, j] = np.max(input[r_start:r_end, c_start:c_end])

    return output
```

PyTorch 中有内置实现：`torch.nn.FractionalMaxPool2d`。

#### 6.2.5 为什么 Fractional Pooling 是正则化

**效果一：多尺度表示**

因为每次池化的切分位置不同，同一个区域的信息在不同前向传播中可能被分到不同的输出位置。模型被迫学到**对空间位置不敏感**的特征，而不是死记特定位置的特定模式。

**效果二：隐式数据增强**

随机切分等价于对中间特征图做了微小的空间抖动（jittering）。类似于数据增强中对图像做微小位移，但作用在特征层面。

**效果三：打破池化的对称性**

标准池化的窗口是严格对齐的，如果输入有周期性噪声，池化可能会系统性地放大这种噪声。随机切分打破了这种对称性。

#### 6.2.6 标准池化 vs Fractional Pooling 对比

| | Max Pooling | Random Pooling | Fractional Max Pooling |
|---|---|---|---|
| 确定性 | 完全确定 | 随机 | 随机 |
| 下采样比例 | 整数倍 | 整数倍 | 任意比例 |
| 正则化效果 | 无 | 弱 | 较强 |
| 计算开销 | 小 | 略大（需采样） | 略大（需生成切分点） |
| 实际使用 | 非常广泛 | 少见 | 少见 |

#### 6.2.7 实际使用情况

Fractional Pooling 在实践中使用频率相对较低，原因：
1. 现代网络更多依赖卷积层的 stride 来控制分辨率，而不是显式池化层
2. Dropout 和 BN 已经提供了足够的正则化
3. 数据增强（如随机裁剪、随机翻转）在某种程度上起到了类似的作用

但在以下场景仍有价值：
- 输入尺寸不固定时，需要灵活的下采样比例
- 希望在中间层引入空间随机性
- 池化层的感受野对最终结果非常敏感时

---

### 6.3 Stochastic Depth vs Fractional Pooling vs Dropout 对比

| | Dropout | Stochastic Depth | Fractional Pooling |
|---|---|---|---|
| **随机化的对象** | 单个神经元 | 整个残差块 | 空间池化区域 |
| **粒度** | 细（神经元级） | 粗（层/块级） | 结构（空间级） |
| **作用阶段** | 全连接层、Attention 层 | 深层残差网络 | 卷积网络中的池化层 |
| **正则化机制** | 防止神经元共适应 | 训练隐式集成 + 减少冗余 | 空间抖动 + 多尺度表示 |
| **额外收益** | — | 减少训练时间 | 支持非整数下采样 |
| **现代代表** | 几乎所有模型 | DropPath (ViT, Swin) | 较少使用 |

**统一理解**：三者的共同哲学是**用随机性打破确定性计算路径**，迫使模型学到更鲁棒、不依赖特定路径或特定位置的表示。

---

## 7. 深度思考："Batch" 的两种含义——梯度下降中的 batch vs 归一化中的 Batch

"Batch" 在深度学习中出现了两种截然不同的语境：

1. **梯度下降中的 batch**：决定"一次更新看多少数据"
2. **归一化中的 Batch**：决定"按什么维度统计均值和方差"

两者名字相近，但解决的问题完全不同。下面分别详解。

---

### 7.1 梯度下降中的 Batch：一次更新看多少数据

#### 7.1.1 核心问题

梯度下降需要计算损失函数对参数的梯度：

$$\nabla_W L = \frac{\partial}{\partial W} \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i, W), y_i)$$

问题是：$N$（训练集大小）可能非常大（百万级）。遍历全部数据才做一次参数更新太慢了。怎么折中？

#### 7.1.2 三种策略

**策略一：Batch Gradient Descent（批梯度下降）**

每次用**全部** $N$ 个样本计算梯度：

$$W_{t+1} = W_t - \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla_W \ell_i$$

- 梯度是**真实梯度**（精确）
- 每次更新的计算量最大（$O(N)$）
- 每次更新方向最可靠，但更新次数最少
- 大数据集上几乎不可用（一次更新可能需要几小时）

**策略二：Stochastic Gradient Descent（随机梯度下降）**

每次只用**1 个**样本：

$$W_{t+1} = W_t - \alpha \cdot \nabla_W \ell_{i_t}$$

其中 $i_t$ 是随机选取的样本索引。

- 梯度是**真实梯度的无偏估计**（期望值相同，但方差很大）
- 计算最快（$O(1)$）
- 但噪声太大，训练曲线剧烈震荡，难以收敛
- 现在几乎不使用纯 SGD

**策略三：Mini-batch Gradient Descent（小批梯度下降）——实践中最常用**

每次用 $B$ 个样本（$B$ 通常为 32、64、128、256）：

$$W_{t+1} = W_t - \alpha \cdot \frac{1}{B} \sum_{i \in \text{batch}} \nabla_W \ell_i$$

- 梯度方差介于 Batch GD 和 SGD 之间
- 计算量适中（$O(B)$）
- 可以利用 GPU 的并行计算能力（矩阵运算对 batch 维度高度并行）

#### 7.1.3 Batch Size 的影响

| Batch Size | 梯度噪声 | 训练速度（每步） | 泛化能力 | 内存占用 |
|---|---|---|---|---|
| 小（1-32） | 大 | 慢（每步快但需要更多步） | **好**（噪声有正则化效果） | 小 |
| 中（64-256） | 适中 | 适中 | 适中 | 适中 |
| 大（512-4096） | 小 | 快（每步慢但总步数少） | **差**（容易过拟合） | 大 |

**小 batch size 为什么泛化更好？**

梯度噪声相当于给优化过程添加了随机扰动，效果类似于：
- 防止模型陷入尖锐的局部最优
- 帮助模型找到更平坦的最优解（flat minima），而平坦的最优解泛化能力更强
- 这是一种**免费的隐式正则化**

这也是为什么 BERT、GPT 等大模型训练时虽然受限于显存而用较大 batch，但仍然通过各种手段（如 warmup、layer-wise lr decay）来弥补大 batch 带来的泛化损失。

#### 7.1.4 小结

```
数据量 N = 100,000

Batch GD:       每步看 100,000 个样本 → 1 步/epoch
SGD:            每步看 1 个样本       → 100,000 步/epoch
Mini-batch 64:  每步看 64 个样本      → 1,562 步/epoch
Mini-batch 256: 每步看 256 个样本     → 390 步/epoch
```

实践中几乎总是使用 mini-batch，因为它是计算效率和泛化能力的最佳折中。

---

### 7.2 归一化中的 Batch：按什么维度统计均值和方差

#### 7.2.1 为什么需要归一化

在深层网络中，随着训练的进行，每层输入的分布会不断变化——这就是所谓的**内部协变量偏移（Internal Covariate Shift）**。

后果：
- 后层需要不断适应前层分布的变化，学习率必须设得很小
- 某些层的输入值可能变得很大或很小，导致梯度消失或爆炸
- 激活函数（如 Sigmoid）在输入值偏离零中心时梯度很小

归一化的目标：**将每层的输入约束到一个合理的范围内（均值 0，方差 1），使训练更稳定。**

#### 7.2.2 归一化方法全景

归一化方法的核心区别在于：**在哪个维度上计算均值和方差？**

对于一个 4D 张量 $X \in \mathbb{R}^{N \times C \times H \times W}$（batch × channel × height × width）：

```
N: 样本数量（batch size）
C: 通道数（channel / feature dimension）
H: 高度（spatial height）
W: 宽度（spatial width）
```

#### 7.2.3 Batch Normalization（批归一化，2015）

**归一化维度**：对每个 channel，在 (N, H, W) 三个维度上统计均值和方差。

$$\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} X_{n,c,h,w}$$

$$\sigma_c^2 = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{h=1}^{H} \sum_{w=1}^{W} (X_{n,c,h,w} - \mu_c)^2$$

$$\hat{X}_{n,c,h,w} = \frac{X_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}$$

$$Y_{n,c,h,w} = \gamma_c \hat{X}_{n,c,h,w} + \beta_c$$

```
N
│  ┌──────────────────────────┐
│  │ 对每个 channel c:          │
│  │   μ_c, σ_c 跨 N×H×W 计算  │
│  │   同一 channel 内所有样本   │
│  │   所有空间位置共享统计量    │
├──┤                          │
│C │  每个 channel 有独立的      │
│  │  μ, σ, γ, β              │
├──┤                          │
│H │                          │
│W │                          │
└──┴──────────────────────────┘
```

**特点**：
- 依赖 batch size：batch 太小（< 8）时统计量不稳定，效果急剧下降
- 训练和测试行为不同：训练用 mini-batch 统计量，测试用 running mean/var
- 副产品正则化：mini-batch 统计量的随机性产生正则化效果（见第4节）
- CV 中广泛使用（ResNet、VGG 等几乎所有 CNN）

**局限**：
- batch size 小时表现差（NLP 中 batch 通常较小）
- 序列长度不固定时难以应用
- 训练和测试的行为差异需要额外的 running statistics

#### 7.2.4 Layer Normalization（层归一化，2016）

**归一化维度**：对每个样本，在 (C, H, W) 三个维度上统计均值和方差。

$$\mu_n = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^{C} \sum_{h=1}^{H} \sum_{w=1}^{W} X_{n,c,h,w}$$

$$\hat{X}_{n,c,h,w} = \frac{X_{n,c,h,w} - \mu_n}{\sqrt{\sigma_n^2 + \epsilon}}$$

$$Y_{n,c,h,w} = \gamma_c \hat{X}_{n,c,h,w} + \beta_c$$

```
N
│  ┌──────────────────────────┐
│  │ 对每个样本 n:              │
│  │   μ_n, σ_n 跨 C×H×W 计算  │
│  │   单个样本内所有特征        │
│  │   所有空间位置共享统计量    │
├──┤                          │
│C │  每个 channel 有独立的      │
│  │  γ, β（但 μ, σ 共享）     │
├──┤                          │
│H │                          │
│W │                          │
└──┴──────────────────────────┘
```

**特点**：
- 不依赖 batch size（每个样本独立归一化）
- 训练和测试行为一致（不需要 running statistics）
- NLP / Transformer 的标准选择
- 正则化效果比 BN 弱（因为没有 mini-batch 随机性）

**为什么 Transformer 用 LayerNorm 而不是 BatchNorm**：
1. NLP 中序列长度不固定，BN 的空间维度统计量不稳定
2. NLP 中 batch size 通常较小（受限于显存和长序列），BN 效果差
3. 自回归生成（如 GPT）时每个时间步只有一个样本，BN 完全无法使用

#### 7.2.5 Instance Normalization（实例归一化，2016）

**归一化维度**：对每个样本的每个 channel，在 (H, W) 上统计。

$$\mu_{n,c} = \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} X_{n,c,h,w}$$

```
N
│  ┌──────────────────────────┐
│  │ 对每个样本 n、每个 channel c:│
│  │   μ_{n,c}, σ_{n,c}       │
│  │   跨 H×W 计算              │
├──┤                          │
│C │  每个 channel 有独立的      │
│  │  γ, β（μ, σ 也独立）     │
├──┤                          │
│H │                          │
│W │                          │
└──┴──────────────────────────┘
```

**特点**：
- 每个 (样本, 通道) 对有独立的统计量
- 彻底去除了样本间的统计量共享
- 主要用于**风格迁移**（style transfer）：因为风格特征（对比度、亮度）在不同空间位置是一致的，IN 可以有效去除风格信息，只保留内容信息
- 一般场景下不如 BN 和 LN

#### 7.2.6 Group Normalization（组归一化，2018）

**归一化维度**：将 channels 分成 $G$ 组，对每组在 (N, H, W) 上统计。

$$\mu_{n,g} = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in \text{group } g} \sum_{h=1}^{H} \sum_{w=1}^{W} X_{n,c,h,w}$$

```
N
│  ┌──────────────────────────┐
│  │ 对每个样本 n、每个组 g:    │
│  │   μ_{n,g}, σ_{n,g}       │
│  │   跨组内 channels × H × W  │
├──┤                          │
│C │  ┌─────┬─────┬─────┐    │
│  │  │Grp 1│Grp 2│ ... │ G个组│
│  │  │c1~c4│c5~c8│     │    │
│  │  └─────┴─────┴─────┘    │
├──┤                          │
│H │                          │
│W │                          │
└──┴──────────────────────────┘
```

**关键观察**：其他归一化方法都是 Group Normalization 的特例：

| $G$（组数） | 等价于 |
|---|---|
| $G = 1$ | **Layer Normalization**（所有通道一组） |
| $G = C$ | **Instance Normalization**（每个通道一组） |
| $G = N$ (batch size) | 类似 **Batch Normalization**（但在 batch 维度分组） |

**特点**：
- 不依赖 batch size（在组内统计，不跨样本）
- CV 中 batch size 较小时（如目标检测、分割任务）的 BN 替代方案
- 典型设置：$G = 32$
- Facebook 的 DETR、Mask R-CNN 等任务中使用

#### 7.2.7 RMS Normalization（均方根归一化，2019）

LayerNorm 的简化版：**只除以均方根，不减均值。**

$$\hat{X}_{n,c,h,w} = \frac{X_{n,c,h,w}}{\text{RMS}_n} = \frac{X_{n,c,h,w}}{\sqrt{\frac{1}{d}\sum_{i} X_{n,i}^2 + \epsilon}}$$

$$Y_{n,c,h,w} = \gamma_c \hat{X}_{n,c,h,w}$$

- 没有偏移参数 $\beta$（只有缩放参数 $\gamma$）
- 计算量更小（不需要减均值）
- LLaMA、GPT-NeoX 等现代大语言模型使用

#### 7.2.8 五种归一化方法统一对比

| 方法 | 统计量计算维度 | 依赖 batch | 训练/测试一致 | 主要场景 |
|---|---|---|---|---|
| **Batch Norm** | (N, H, W) per channel | 是 | 否（需要 running stats） | CNN（ResNet 等） |
| **Layer Norm** | (C, H, W) per sample | 否 | 是 | Transformer、NLP |
| **Instance Norm** | (H, W) per (sample, channel) | 否 | 是 | 风格迁移 |
| **Group Norm** | (C/G, H, W) per (sample, group) | 否 | 是 | 小 batch 的 CV 任务 |
| **RMS Norm** | (C, H, W) per sample（只算方差） | 否 | 是 | LLaMA 等大模型 |

```
归一化维度从大到小：

Batch Norm:    ████████████████████  跨 N×H×W（最粗，依赖 batch）
Group Norm:    ████████████████      跨 (C/G)×H×W
Layer Norm:    ████████████          跨 C×H×W
Instance Norm: ████████              跨 H×W（最细，完全独立）
```

#### 7.2.9 一个直觉总结

所有归一化方法都在做同一件事：**让数据保持在"好"的范围内，使优化更稳定。**

区别只在于"跟谁比"：
- Batch Norm：跟同一个 batch 里其他样本的统计量比 → CV 主流
- Layer Norm：跟自己所有通道的统计量比 → NLP 主流
- Instance Norm：跟自己单个通道的统计量比 → 风格迁移
- Group Norm：跟自己一组通道的统计量比 → 小 batch 的折中

---

### 7.3 "Batch" 两种含义的关系

| | 梯度下降中的 batch | 归一化中的 Batch Norm |
|---|---|---|
| 控制什么 | 一次参数更新看多少数据 | 归一化时跨多少样本统计 |
| 主要影响 | 训练速度 + 泛化能力 | 训练稳定性 + 正则化 |
| batch size 太小 | 训练慢但泛化好 | BN 统计量不稳定，效果差 |
| batch size 太大 | 训练快但泛化差 | BN 效果好，但缺少正则化 |
| 联系 | 小 batch 的梯度噪声本身就是隐式正则化 | BN 的 mini-batch 噪声也是隐式正则化 |

**一个有趣的矛盾**：

- 梯度下降角度：小 batch 更好（泛化好）
- Batch Norm 角度：大 batch 更好（统计量更稳定）

实践中需要权衡。现代常见做法：
- CV：batch = 32~256，用 BN
- NLP：batch 较小或用 gradient accumulation，用 LN / RMS Norm

---

## 8. 数学深入：Hessian 矩阵与随机近似算法的收敛性分析

---

### 8.1 Hessian 矩阵：梯度之外的"二阶信息"

#### 8.1.1 从梯度到 Hessian

梯度告诉我们函数在某个点"往哪个方向走最陡"，但它只提供了**一阶信息**（斜率）。Hessian 矩阵提供的是**二阶信息**（曲率）——不仅告诉我们方向，还告诉我们每个方向的"弯曲程度"。

对于一个标量函数 $f: \mathbb{R}^d \to \mathbb{R}$（比如损失函数 $L(W)$）：

- **梯度（一阶导数）**：$g = \nabla f = \left[\frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_d}\right]^T \in \mathbb{R}^d$
- **Hessian 矩阵（二阶导数）**：$H = \nabla^2 f \in \mathbb{R}^{d \times d}$，其中

$$H_{ij} = \frac{\partial^2 f}{\partial w_i \partial w_j}$$

Hessian 是一个**对称矩阵**（当 $f$ 二阶连续可微时），因为 $\frac{\partial^2 f}{\partial w_i \partial w_j} = \frac{\partial^2 f}{\partial w_j \partial w_i}$。

#### 8.1.2 二阶 Taylor 展开

Hessian 的数学意义最直观地体现在二阶 Taylor 展开中。在点 $W$ 附近展开：

$$f(W + \Delta W) \approx f(W) + \nabla f(W)^T \Delta W + \frac{1}{2} \Delta W^T \nabla^2 f(W) \Delta W$$

对比一阶展开：

$$f(W + \Delta W) \approx f(W) + \nabla f(W)^T \Delta W \quad \text{（只用梯度，是线性近似）}$$

$$f(W + \Delta W) \approx f(W) + \nabla f(W)^T \Delta W + \frac{1}{2} \Delta W^T H \Delta W \quad \text{（加上 Hessian，是二次近似）}$$

多出来的 $\frac{1}{2} \Delta W^T H \Delta W$ 这一项就是曲率信息。它告诉我们：
- 如果 $\Delta W^T H \Delta W > 0$：函数在这个方向是"凹的"（碗状）
- 如果 $\Delta W^T H \Delta W < 0$：函数在这个方向是"凸的"（山脊状）

#### 8.1.3 Hessian 的特征值与损失面的几何

对 Hessian 做特征值分解：$H = V \Lambda V^T$，其中 $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_d)$。

特征值 $\lambda_i$ 的含义：

| 特征值 | 对应方向的曲率 | 直觉 |
|---|---|---|
| $\lambda_i > 0$（正） | 凹方向（碗底） | 沿特征向量 $v_i$ 方向移动，函数值上升 |
| $\lambda_i < 0$（负） | 凸方向（山脊） | 沿特征向量 $v_i$ 方向移动，函数值下降 |
| $\lambda_i = 0$（零） | 平坦方向 | 函数在这个方向上不变化（鞍点） |
| $\lambda_i \gg 1$（大） | 陡方向 | 函数变化剧烈 |
| $\lambda_i \ll 1$（小） | 浅方向 | 函数变化缓慢 |

根据特征值的符号，可以判断驻点的类型：

- **所有 $\lambda_i > 0$**：Hessian 正定 → **局部最小值**（碗底）
- **所有 $\lambda_i < 0$**：Hessian 负定 → **局部最大值**（山顶）
- **$\lambda_i$ 有正有负**：Hessian 不定 → **鞍点**（某些方向是碗底，某些方向是山脊）

**高维空间中鞍点远比局部最小值常见**。直观上：$d$ 维空间中，Hessian 的 $d$ 个特征值全部为正的概率很低，只要有一个为负就不是局部最小值。维度越高，鞍点越多。

#### 8.1.4 条件数与优化难度

Hessian 的**条件数**（condition number）定义为：

$$\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

条件数衡量损失面的"椭球形状有多扁"：

```
条件数 κ ≈ 1（各方向曲率均匀）：    条件数 κ ≫ 1（各方向曲率差异大）：

     ╱╲                                  ╱╲
    ╱  ╲                                ╱    ╲
   ╱    ╲                             ╱────────╲
  ╱      ╲                           ╱          ╲
 ╱────────╲                          ╱────────────╲
   (圆形)                              (狭长椭圆)
 梯度下降可以快速收敛                  梯度下降在狭长方向上反复震荡
```

当 $\kappa$ 很大时：
- 在 $\lambda_{\max}$ 对应的**陡方向**：梯度大，更新步长 $\alpha g$ 也大，但容易"overshoot"，需要很小学习率才能不发散
- 在 $\lambda_{\min}$ 对应的**浅方向**：梯度小，更新步长 $\alpha g$ 也小，收敛极慢
- **结果**：无论学习率设大设小，都有一个方向表现不好

这解释了为什么课件中说（第 119 页）：在浅方向（shallow）上进展缓慢，在陡方向（steep）上波动剧烈。

#### 8.1.5 最优学习率与 Hessian

对于二次函数 $f(W) = \frac{1}{2} W^T H W - b^T W$，梯度下降的最优学习率有解析解：

$$\alpha^* = \frac{2}{\lambda_{\max} + \lambda_{\min}}$$

收敛速率为：

$$\left(\frac{\kappa - 1}{\kappa + 1}\right)^t \to 0 \quad \text{当 } t \to \infty$$

当 $\kappa = 1$ 时一步收敛；$\kappa = 100$ 时需要约 100 步；$\kappa = 10000$ 时需要约 10000 步。

**这就是为什么需要自适应学习率（RMSProp、Adam）**：它们在 Hessian 曲率大的方向自动减小步长，在曲率小的方向自动增大步长，等价于在隐式地"预调节"（precondition）Hessian。

#### 8.1.6 Newton 法：利用 Hessian 的理想方法

如果直接利用 Hessian（而不仅仅是梯度），可以得到 Newton 法：

$$W_{t+1} = W_t - H^{-1} \nabla f(W_t)$$

Newton 法的问题：
1. 计算 Hessian 的成本：$O(d^2)$ 存储和 $O(d^3)$ 求逆
2. 深度学习中 $d$ 可能是数十亿，完全不可行
3. Hessian 不定时（有鞍点），$H^{-1}$ 可能不存在或导致方向错误

**现代优化器的本质就是在近似 Newton 法**：

| 方法 | 隐式近似 $H^{-1}$ 的方式 |
|---|---|
| SGD | 用单位矩阵 $I$ 近似 $H^{-1}$（完全忽略曲率） |
| Momentum | 用一阶累积近似曲率的对角线 |
| RMSProp | 用梯度平方的累积近似 $H$ 的对角线，然后取倒数 |
| Adam | Momentum + RMSProp = 一阶 + 二阶混合近似 |

#### 8.1.7 1D 和 2D 的直观例子

**一维**：$f(x) = x^2$

- 梯度：$f'(x) = 2x$
- Hessian：$f''(x) = 2$（标量，正数 = 碗状）
- 条件数：$\kappa = 2/2 = 1$（完美，一步收敛）

**一维**：$f(x) = x^2 + 10x^4$

- 梯度：$f'(x) = 2x + 40x^3$
- Hessian：$f''(x) = 2 + 120x^2$
- 在 $x = 0$：$f''(0) = 2 > 0$，局部最小值
- 在 $x = 1$：$f''(1) = 122 > 0$，更陡
- 在 $x = 10$：$f''(10) = 12002$，极陡

**二维**：$f(x_1, x_2) = \lambda_1 x_1^2 + \lambda_2 x_2^2$

$$H = \begin{bmatrix} 2\lambda_1 & 0 \\ 0 & 2\lambda_2 \end{bmatrix}$$

若 $\lambda_1 = 100, \lambda_2 = 1$：
- $x_1$ 方向极陡（100 倍），$x_2$ 方向极浅
- 条件数 $\kappa = 100$
- 梯度下降会在 $x_1$ 方向反复震荡，在 $x_2$ 方向爬行

这就是课件第 117-119 页所描述的现象。

---

### 8.2 随机近似算法（Stochastic Approximation）的收敛性证明

#### 8.2.1 问题的精确数学表述

随机梯度下降是**随机近似**（Robbins-Monro, 1951）算法的一个特例。

**确定性问题**：

$$W_{t+1} = W_t - \alpha \nabla F(W_t), \quad F(W) = \frac{1}{N}\sum_{i=1}^N f_i(W)$$

**随机近似问题**：

$$W_{t+1} = W_t - \alpha_t \nabla f_{i_t}(W_t)$$

其中 $i_t$ 是第 $t$ 步随机抽取的样本索引。定义噪声项：

$$\xi_t = \nabla f_{i_t}(W_t) - \nabla F(W_t)$$

则更新公式等价于：

$$W_{t+1} = W_t - \alpha_t \left[\nabla F(W_t) + \xi_t\right]$$

这里 $\xi_t$ 是**零均值随机噪声**：$\mathbb{E}[\xi_t | W_t] = 0$。

#### 8.2.2 Robbins-Monro 条件：为什么 $\sum \alpha_t = \infty$ 且 $\sum \alpha_t^2 < \infty$

Robbins 和 Monro 在 1951 年证明了，如果要使随机近似算法收敛到方程的根 $W^*$（即 $\nabla F(W^*) = 0$），学习率序列 $\{\alpha_t\}$ 必须满足：

$$\boxed{\sum_{t=0}^{\infty} \alpha_t = \infty} \qquad \text{且} \qquad \boxed{\sum_{t=0}^{\infty} \alpha_t^2 < \infty}$$

**条件一的证明直觉：$\sum \alpha_t = \infty$（保证能走到最优解）**

如果 $\sum \alpha_t < \infty$，意味着学习率的"总预算"是有限的。即使每一步都精确地朝着最优解方向走，总移动距离也是有限的：

$$\sum_{t=0}^{\infty} \|\alpha_t \nabla F(W_t)\| \leq \sum_{t=0}^{\infty} \alpha_t \cdot G < \infty$$

其中 $G$ 是梯度的上界。这意味着如果初始点离最优解足够远，算法可能根本走不到最优解。

类比：你从上海出发去北京，每天最多走 $\alpha_t$ 公里。如果总步数有限（$\sum \alpha_t < \infty$），你可能永远到不了北京。

**条件二的证明直觉：$\sum \alpha_t^2 < \infty$（保证噪声不会累积）**

每一步的更新中包含噪声项 $\xi_t$，噪声造成的位移为 $\alpha_t \xi_t$。噪声的方差（累积效应）为：

$$\text{Var}\left(\sum_{t=0}^{T} \alpha_t \xi_t\right) = \sum_{t=0}^{T} \alpha_t^2 \text{Var}(\xi_t) \leq \sigma^2 \sum_{t=0}^{T} \alpha_t^2$$

如果 $\sum \alpha_t^2 < \infty$，噪声的累积方差有上界，不会随时间发散。即：虽然每一步都有随机扰动，但扰动的总影响是可控的。

类比：你每天走路时有一个随机的侧向偏移 $\alpha_t \xi_t$。如果偏移的平方和收敛，你最终虽然不在精确的路线上，但偏移不会无限增大。

**两个条件的矛盾与统一**：

- 条件一要求学习率不能衰减太快（否则走不到）
- 条件二要求学习率必须衰减足够快（否则噪声失控）
- $\alpha_t = \frac{1}{t}$ 恰好同时满足：$\sum 1/t = \infty$（调和级数发散）且 $\sum 1/t^2 < \infty$（p-级数收敛）

**常见满足条件的学习率序列**：

| 学习率 | $\sum \alpha_t$ | $\sum \alpha_t^2$ | 满足 RM 条件？ |
|---|---|---|---|
| $\alpha_t = \frac{1}{t}$ | $\infty$（调和级数） | $\pi^2/6$（收敛） | 满足 |
| $\alpha_t = \frac{1}{\sqrt{t}}$ | $\infty$ | $\infty$ | **不满足**（噪声发散） |
| $\alpha_t = \frac{1}{t^{0.6}}$ | $\infty$ | $\infty$ | **不满足**（噪声发散） |
| $\alpha_t = \frac{1}{t^{1.1}}$ | 有限 | 有限 | **不满足**（走不到） |
| $\alpha_t = c$（常数） | $\infty$ | $\infty$ | **不满足**（噪声发散） |

#### 8.2.3 随机梯度下降的收敛定理（Lipschitz + 强凸）

**定理**（SGD 收敛性）：

设 $F(W)$ 满足以下条件：
1. **$L$-Lipschitz 连续梯度**：$\|\nabla F(W) - \nabla F(W')\| \leq L \|W - W'\|$
2. **$\mu$-强凸**：$F(W) \geq F(W^*) + \nabla F(W^*)^T(W - W^*) + \frac{\mu}{2}\|W - W^*\|^2$
3. 噪声有界：$\mathbb{E}[\|\xi_t\|^2] \leq \sigma^2$

如果使用 Robbins-Monro 学习率 $\alpha_t = \frac{1}{\mu t}$，则：

$$\mathbb{E}[F(\bar{W}_T) - F(W^*)] \leq O\left(\frac{L \sigma^2 \cdot \log T}{\mu^2 T}\right)$$

其中 $\bar{W}_T = \frac{1}{T}\sum_{t=1}^T W_t$ 是迭代点的平均。

**证明框架**：

**Step 1**：利用 $L$-Lipschitz 条件和 $W_{t+1}$ 的更新公式，建立下降不等式。

由 $L$-Lipschitz 连续梯度，有（descent lemma）：

$$F(W_{t+1}) \leq F(W_t) + \nabla F(W_t)^T (W_{t+1} - W_t) + \frac{L}{2}\|W_{t+1} - W_t\|^2$$

代入 $W_{t+1} - W_t = -\alpha_t (\nabla F(W_t) + \xi_t)$：

$$F(W_{t+1}) \leq F(W_t) - \alpha_t \|\nabla F(W_t)\|^2 - \alpha_t \nabla F(W_t)^T \xi_t + \frac{L\alpha_t^2}{2} \|\nabla F(W_t) + \xi_t\|^2$$

**Step 2**：对 $W^*$ 的距离建立递推。

利用 $\mu$-强凸条件：$\|\nabla F(W_t)\|^2 \geq 2\mu (F(W_t) - F(W^*))$，展开并取期望（注意到 $\mathbb{E}[\xi_t | W_t] = 0$）：

$$\mathbb{E}[\|W_{t+1} - W^*\|^2] \leq \mathbb{E}[\|W_t - W^*\|^2] - 2\alpha_t \mu \cdot \mathbb{E}[F(W_t) - F(W^*)] + L\alpha_t^2 (G^2 + \sigma^2)$$

其中 $G$ 是梯度的上界。

**Step 3**：递推展开（telescoping sum）。

将上式从 $t = 0$ 到 $T-1$ 求和：

$$\sum_{t=0}^{T-1} 2\alpha_t \mu \cdot \mathbb{E}[F(W_t) - F(W^*)] \leq \|W_0 - W^*\|^2 + L(G^2 + \sigma^2) \sum_{t=0}^{T-1} \alpha_t^2$$

**Step 4**：利用 $\sum \alpha_t = \infty$ 和 $\sum \alpha_t^2 < \infty$ 的条件。

代入 $\alpha_t = \frac{1}{\mu t}$，利用 $\sum_{t=1}^T \frac{1}{t} \approx \log T$ 和 $\sum_{t=1}^T \frac{1}{t^2} \leq \frac{\pi^2}{6}$，得到：

$$\sum_{t=1}^{T} \frac{2}{t} \cdot \mathbb{E}[F(W_t) - F(W^*)] \leq \|W_0 - W^*\|^2 + \frac{L(G^2 + \sigma^2)\pi^2}{6\mu^2}$$

因为左边求和中的项 $\frac{2}{t} \cdot \mathbb{E}[F(W_t) - F(W^*)] \geq 0$，且求和的权重 $\sum \frac{2}{t} \approx 2\log T$ 发散，所以必须有无穷多项 $\mathbb{E}[F(W_t) - F(W^*)] \to 0$，即算法收敛。

更精确地，用 $\bar{W}_T = \frac{1}{T}\sum_{t=1}^T W_t$ 的时间平均，最终得到：

$$\mathbb{E}[F(\bar{W}_T) - F(W^*)] = O\left(\frac{\log T}{T}\right)$$

**证毕。**

#### 8.2.4 常数学习率：理论与实践的鸿沟

**理论要求** $\alpha_t \to 0$（如 $\alpha_t = 1/t$），但**实践中几乎总是用常数学习率**（如 $\alpha = 0.001$）。为什么？

**原因一：非凸性与深度学习的现实**

上述收敛定理假设 $F$ 是 $\mu$-强凸的——即损失面是一个漂亮的"碗"。但深度学习的损失面：
- 是非凸的（有大量鞍点和局部最优）
- Hessian 的条件数极大（$10^3 \sim 10^6$）
- 有效维度可能远低于参数维度（flat directions）

在非凸场景下，SGD 的收敛定理变为：

$$\mathbb{E}[F(\bar{W}_T) - F(W^*)] \leq O\left(\frac{1}{\sqrt{T}}\right) \quad \text{（非凸 SGD，常数学习率）}$$

常数学习率在非凸情况下**已经可以保证收敛到平稳点**（$\mathbb{E}[\|\nabla F(W_t)\|^2] \to 0$），不需要衰减到零。

**原因二：常数学习率的"噪声有益"效应**

$$W_{t+1} = W_t - \alpha \nabla F(W_t) - \alpha \xi_t$$

最后一项 $-\alpha \xi_t$ 是梯度噪声。它有两个效果：
1. **逃离鞍点**：在鞍点处梯度为零，但噪声可以把参数推出鞍点，让优化继续
2. **倾向平坦最优解**：噪声使参数在最优解附近震荡，最终倾向于停留在"宽"的区域（因为"窄"的区域中噪声容易把参数推出）

这正是泛化能力好的原因。如果学习率衰减到零，噪声也衰减到零，就失去了这些好处。

**原因三：常数学习率 + 学习率调度是更好的折中**

实践中不是完全不变的学习率，而是：
1. 先用 warmup 把学习率升上去
2. 然后保持较长时间
3. 最后用 cosine decay / step decay 衰减

这相当于：前期利用噪声探索（找好的区域），后期用小学习率精调（收敛到好的解）。

**数学上的等价理解**：

常数学习率 $\alpha$ 可以理解为"等效于" Robbins-Monro 条件下的**mini-batch**随机梯度下降。

对于 mini-batch SGD：

$$W_{t+1} = W_t - \alpha \cdot \frac{1}{B} \sum_{i \in \text{batch}} \nabla f_i(W_t)$$

噪声的方差为 $\text{Var}(\xi_t) = \sigma^2 / B$。当 $B \to N$（用全部数据）时，噪声消失，$\alpha$ 可以取常数。因此：

$$\boxed{\text{增大 batch size} \approx \text{减小等效噪声} \approx \text{允许更大的学习率}}$$

这解释了实践中的经验法则：batch size 增大 $N$ 倍时，学习率也按 $N$ 比例缩放。但"为什么"需要更深入的分析。

##### Batch Size 与学习率关系的本质推导

**起点：mini-batch 梯度的统计性质**

设真实梯度为 $\nabla F(W) = \frac{1}{N}\sum_{i=1}^N \nabla f_i(W)$，mini-batch 梯度为：

$$g_t = \frac{1}{B}\sum_{i \in \text{batch}} \nabla f_i(W_t)$$

将 $g_t$ 分解为**信号 + 噪声**：

$$g_t = \underbrace{\nabla F(W_t)}_{\text{信号（真实梯度）}} + \underbrace{\xi_t}_{\text{噪声}}$$

其中噪声 $\xi_t$ 的统计性质：

$$\mathbb{E}[\xi_t] = 0, \qquad \text{Var}(\xi_t) = \frac{\sigma^2}{B}$$

每一步的更新为：

$$W_{t+1} = W_t - \alpha \cdot g_t = W_t - \alpha \nabla F(W_t) - \alpha \xi_t$$

**问题：当我们把 batch size 从 $B$ 增大到 $kB$ 时，$\alpha$ 应该怎么变？**

**推导一：每个 epoch 的总更新量不变（线性缩放的真正理由）**

考虑一个 epoch（遍历全部 $N$ 个样本一次），计算总的参数更新。

- 每步更新：$\Delta W = -\alpha \cdot g_t$
- 一个 epoch 内的步数：$N / B$
- 一个 epoch 内的总更新（只看信号部分，忽略噪声）：

$$\text{总信号} = \sum_{\text{epoch}} \alpha \cdot \nabla F(W) \approx \frac{N}{B} \cdot \alpha \cdot \nabla F(W)$$

**如果要使"每个 epoch 的总更新量"与 batch size 无关**，则需要：

$$\frac{N}{B} \cdot \alpha = \text{const}$$

$$\boxed{\alpha \propto B}$$

这就是线性缩放规则。它的含义非常直观：**batch 大了，每个 epoch 的步数少了，所以每步要迈更大才跟得上。**

类比：你每天跑 $\alpha$ 公里，一个月（epoch）跑 $N/B$ 天。如果从每周跑 4 天变成每周跑 1 天（$B$ 增大 4 倍），你要每天跑 4 倍远（$\alpha$ 增大 4 倍），才能在一个月内跑同样的总距离。

**推导二：每个 epoch 的总噪声不变（线性缩放的噪声视角）**

再来看噪声。一个 epoch 内，噪声引起的总位移的方差为：

$$\text{总噪声方差} = \sum_{t=1}^{N/B} \text{Var}(\alpha \xi_t) = \frac{N}{B} \cdot \alpha^2 \cdot \frac{\sigma^2}{B} = \frac{N \alpha^2 \sigma^2}{B^2}$$

代入线性缩放 $\alpha = cB$：

$$\text{总噪声方差} = \frac{N (cB)^2 \sigma^2}{B^2} = N c^2 \sigma^2$$

**与 $B$ 无关！** 线性缩放同时保持了：
- 每个 epoch 的总信号 $\propto N c$（不变）
- 每个 epoch 的总噪声方差 $\propto N c^2$（不变）

这是线性缩放规则的**最深层的理由**：它使得无论 batch size 多大，每个 epoch 对参数的"有效推动"和"有效扰动"都是一样的。

**推导三：为什么不线性缩放也能训练？——每步的信号噪声比**

如果我们不缩放学习率（$\alpha$ 固定），会怎样？

每步的**信号噪声比（SNR）**：

$$\text{SNR} = \frac{\|\alpha \nabla F(W)\|}{\sqrt{\text{Var}(\alpha \xi_t)}} = \frac{\alpha \|\nabla F(W)\|}{\alpha \sigma / \sqrt{B}} = \frac{\|\nabla F(W)\| \sqrt{B}}{\sigma}$$

- 增大 $B$，SNR 增大（每步更"准"）
- 但每步的信号 $\alpha \nabla F(W)$ 不变（每步走一样远）
- 只是在更少的步数里走完一个 epoch

所以**不缩放学习率，训练不会发散，只是更慢**（因为每个 epoch 的总信号变小了：$N/B \cdot \alpha$ 随 $B$ 增大而减小）。

**推导四：为什么线性缩放在 batch 极大时失效？——噪声的"泛化收益"被杀死**

上面说每个 epoch 的总噪声方差不变。但这恰恰是问题所在。

噪声的"泛化收益"来自每步的随机扰动——它帮助优化器探索不同的区域、逃出鞍点、倾向平坦最优解。

当我们用线性缩放时：
- 每 step 的噪声：$\alpha \xi_t$，方差为 $\alpha^2 \sigma^2 / B = c^2 B \sigma^2 / B = c^2 \sigma^2$（每步噪声恒定！）
- 但每 epoch 的 step 数减少了：$N/B$

这意味着**噪声的"频率"降低了**：每个 epoch 里被噪声"摇晃"的次数少了。虽然总噪声方差不变，但噪声作用于参数的方式从"高频小幅抖动"变成了"低频大幅抖动"。

而泛化受益的恰恰是"高频小幅抖动"——它让优化器持续探索参数空间的不同方向，而不是偶尔被大力推一下。

数学上说：$T$ 步随机游走覆盖的区域正比于 $\sigma \sqrt{T}$。线性缩放下每个 epoch 的步数为 $N/B$，所以覆盖区域 $\propto \sqrt{N/B}$，随 $B$ 增大而**缩小**。这就是大 batch 泛化差的根本原因。

**总结：一张图讲清全部关系**

```
                    Batch Size B 增大
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      每 epoch 步数↓   梯度方差↓     每步噪声方差↓
        N/B            σ²/B         α²σ²/B
            │             │             │
            │      ┌──────┴──────┐      │
            │      ▼             ▼      │
            │   SNR↑        稳定性↑     │
            │      │             │      │
            ▼      │             │      │
      需要更大的α      │             │
      来补偿步数减少     │             │
            │      │             │      │
            ▼      ▼             ▼      │
         线性缩放 α∝B：每 epoch 总信号不变，总噪声不变
            │
            │   但：噪声频率↓ → 探索范围↓ → 泛化↓
            │
            ▼
      折中：平方根缩放 α∝√B，或大 batch + 小 batch 微调
```

一句话概括：

> **线性缩放的本质是"每个 epoch 的总推动量不变"；但它牺牲了"每个 epoch 的噪声探索次数"，所以大 batch 训练快但不泛化，需要用小 batch 微调来弥补。**

---

#### 8.2.4.1 Batch Size 与学习率的经验法则（附分析）

实践中，batch size 的选择与学习率的配合是一门经验科学，背后有坚实的数学支撑。以下是经过大量实验验证的经验规则。

**线性缩放规则（Linear Scaling Rule）**

$$\text{当 batch size 从 } B \text{ 增大到 } kB \text{ 时，学习率从 } \alpha \text{ 增大到 } k\alpha$$

这是最基本也最广泛使用的规则，来自 Facebook (Goyal et al., 2017) 训练 ImageNet 的实验。

**数学解释**：

mini-batch 梯度的方差为 $\text{Var}(g_t) = \sigma^2 / B$。梯度噪声的"信号噪声比"（SNR）为：

$$\text{SNR} = \frac{\|\nabla F(W)\|}{\sigma / \sqrt{B}} = \frac{\|\nabla F(W)\| \sqrt{B}}{\sigma}$$

当 $B$ 增大 $k$ 倍时，SNR 增大 $\sqrt{k}$ 倍。噪声相对减小，因此可以用更大的学习率而不发散。

**但线性缩放只在一定范围内有效**：

| Batch Size | 推荐学习率 | 来源 |
|---|---|---|
| 32 | 0.1（SGD+Momentum） | ResNet 原始设置 |
| 64 | 0.2 | 线性缩放 |
| 128 | 0.4 | 线性缩放 |
| 256 | 0.8 | 线性缩放 |
| 512 | 1.0~1.2 | 线性缩放开始失效 |
| 1024 | 1.0~1.5 | 需要配合 warmup |
| 2048+ | 1.0~2.0 | 需要更长的 warmup + 可能需要 LARS |

当 batch size 超过 512-1024 时，纯线性缩放往往导致性能下降。原因：

1. **噪声有益论**（Smith & Le, 2017）：噪声帮助泛化。大 batch 噪声小，泛化差。线性放大学习率虽然加速了收敛，但不能补偿噪声的损失
2. **最优点偏移**：大 batch 训练会收敛到不同的最优解（更尖锐），即使训练损失相同，测试表现也差

**平方根缩放规则（Square Root Scaling）**

$$\alpha' = \alpha \sqrt{\frac{B'}{B}}$$

这是一个更保守的替代方案。在 batch size 极大（如 8192）时表现比线性缩放更好。

**数学解释**：线性缩放保持"每步的梯度更新幅度与 batch 大小成正比"，而平方根缩放保持"每步的信号噪声比变化与学习率变化匹配"。因为 SNR 只与 $\sqrt{B}$ 成正比，所以学习率也只应该与 $\sqrt{B}$ 成正比。

| 缩放规则 | 公式 | 适用场景 |
|---|---|---|
| 线性 | $\alpha' = \alpha \cdot B'/B$ | $B \leq 512$，配合 warmup |
| 平方根 | $\alpha' = \alpha \cdot \sqrt{B'/B}$ | $B > 512$，或对泛化要求高 |
| 不变 | $\alpha' = \alpha$ | batch size 变化不大时 |

**Warmup：大 batch 的必选项**

无论使用哪种缩放规则，大 batch 训练几乎都需要 warmup：

```python
# 典型的 warmup + cosine decay 调度
def lr_schedule(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        # 线性 warmup：从 0 增长到 base_lr
        return base_lr * step / warmup_steps
    else:
        # cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + cos(pi * progress))
```

为什么大 batch 必须用 warmup？

训练初期，参数是随机初始化的，梯度方向变化剧烈，与真实梯度方向相差很大（初始化阶段 Hessian 条件数极大）。如果一开始就用大学习率，参数会被推到损失面的"坏区域"，之后的训练再也回不来。warmup 让模型先用小学习率找到损失面的"大方向"，再切换到大学习率加速收敛。

**LARS：Layer-wise Adaptive Rate Scaling**

当 batch size 超过 32K（如 Facebook 训练 ResNet-50 用 batch size=32K），即使 warmup + cosine decay 也不够。需要 LARS（You et al., 2017）：

$$\alpha_l = \eta \cdot \frac{\|W_l\|}{\|\nabla F_l(W)\| + \lambda \|W_l\|}$$

每个层的学习率不同——参数范数大的层用更大学习率，梯度范数大的层用更小学习率。这等价于对每一层做自适应的信任域（trust region）控制。

**不同任务中的典型配置**

| 任务 | 典型 Batch Size | 典型学习率 | 优化器 | 特殊处理 |
|---|---|---|---|---|
| 图像分类（ResNet） | 32~256 | 0.1~0.4 | SGD+Momentum | Step decay 或 cosine |
| 目标检测（Faster R-CNN） | 2~16 | 0.01~0.02 | SGD+Momentum | 小 batch，不能用 BN |
| 机器翻译（Transformer） | ~4096 tokens | 0.0003~0.001 | Adam | Inverse sqrt warmup |
| BERT 预训练 | 256~1024 sequences | 2e-4~5e-4 | Adam | Linear warmup 10000 步 |
| GPT-3 预训练 | 3.2M tokens | 6e-5 | Adam | Cosine decay, 375M token warmup |
| LLaMA 预训练 | 4M tokens | 3e-4 | AdamW | Cosine decay, 2000 step warmup |
| Stable Diffusion | 2048~4096 | 1e-4 | AdamW | Warmup 10000 步 |

**关键观察**：
- NLP/多模态任务几乎都用 Adam/AdamW，学习率通常在 1e-4 到 5e-4 量级
- CV 任务（特别是分类）偏好 SGD+Momentum，学习率通常在 0.1 量级
- 预训练的 batch size 远大于微调（预训练需要大 batch 来充分利用 GPU，微调 batch 小则泛化好）

---

#### 8.2.4.2 关于 batch size 的深入介入

上面是"经验"的部分。以下是我认为更值得思考的几个问题。

**第一，大 batch 真的"不好"吗？——取决于你怎么定义"好"**

Keskar et al. (2016) 的著名论文 "On Large-Batch Training for Deep Learning" 提出了一个被广泛引用的结论：

> 大 batch 倾向于收敛到"尖锐"的最优解（sharp minima），泛化差；小 batch 倾向于收敛到"平坦"的最优解（flat minima），泛化好。

这个结论影响极深，但有一个关键问题：**"尖锐"和"平坦"的度量方式本身就值得质疑。**

Dinh et al. (2017) 的反例论文 "Sharp Minima Can Generalize For Deep Networks" 证明：通过简单的参数重参数化（如 $W \to W/2$），可以找到一个"尖锐"的最优解和一个"平坦"的最优解，它们对应完全相同的网络函数、完全相同的测试误差。尖锐与否取决于**参数化的选择**，而不是泛化能力的本质。

所以更准确的表述应该是：

> 大 batch 训练缺少噪声，因此对初始化更敏感、更容易陷入初始参数附近的局部最优（而这些最优未必是泛化最好的）。小 batch 的噪声让优化在参数空间中探索了更大的范围。

**第二，线性缩放规则为什么在 batch 极大时失效？——梯度估计的偏差**

线性缩放的理论基础是：mini-batch 梯度是真实梯度的无偏估计。这个假设在大多数情况下成立，但有一个被忽视的前提：**每个样本的梯度计算是独立的**。

在分布式训练中，当 batch size 超过单个 GPU 的容量时，需要跨 GPU/节点同步梯度。这时：
- 梯度压缩（gradient compression）会引入偏差
- 同步延迟（all-reduce latency）意味着不同 GPU 看到的模型版本可能不完全一致
- 混合精度训练（FP16）的舍入误差在 batch 大时累积更多

这些工程因素都使得"无偏估计"的假设不再严格成立。

**第三，实践中的真正建议——不要迷信任何单一规则**

1. **先定 batch size，再调学习率**。Batch size 通常由显存和训练速度决定（能塞多少数据就塞多少），学习率才是需要调的超参数
2. **batch size 变化不大（< 2 倍）时，不用改学习率**。线性缩放在小范围内几乎是过度操作
3. **batch size 变化大（> 4 倍）时，用平方根缩放作为起点**，然后在此基础上微调
4. **永远用 warmup**。无论 batch 多大，warmup 几乎没有副作用，但能防止训练早期的灾难性发散
5. **大 batch 训练完成后，用小 batch 微调几个 epoch**。这被称为 "large-batch training + small-batch fine-tuning"，兼顾了速度和泛化

#### 8.2.5 SGD 逃逸鞍点的严格分析

**命题**：设 $W_t$ 是一个严格鞍点（Hessian 至少有一个负特征值），梯度噪声 $\xi_t$ 满足 $\mathbb{E}[\|\xi_t\|^2] \leq \sigma^2$。则 SGD 在常数学习率 $\alpha$ 下，经过 $T = O\left(\frac{1}{\alpha^2 \lambda_{-}}\right)$ 步后，以高概率逃出鞍点的邻域。其中 $\lambda_{-}$ 是 Hessian 最大负特征值的绝对值。

**证明思路**：

在鞍点 $W_s$ 附近，沿 Hessian 负特征向量方向 $v$ 展开损失函数：

$$F(W_s + \delta v) \approx F(W_s) + \frac{\lambda_{-}}{2} \delta^2$$

因为 $\lambda_{-} < 0$，沿 $v$ 方向移动会**降低**损失函数。

SGD 更新中噪声项 $-\alpha \xi_t$ 在 $v$ 方向的分量为 $-\alpha \langle \xi_t, v \rangle$。因为 $\xi_t$ 是各向同性的（通常假设），这个分量以常数概率为正（朝有利方向走）。

经过 $T$ 步累积，沿 $v$ 方向的位移近似于随机游走，位移的标准差为 $\alpha \sigma \sqrt{T}$。当这个位移超过鞍点邻域的半径时，SGD 就逃出了鞍点。

$$\alpha \sigma \sqrt{T} \geq r \quad \Longrightarrow \quad T \geq \frac{r^2}{\alpha^2 \sigma^2}$$

这就是为什么：
- **常数学习率是必要的**：如果 $\alpha_t \to 0$，逃逸时间 $T \to \infty$，即永远被困在鞍点
- **噪声是好事**：在非凸优化中，噪声帮助跳出鞍点和尖锐局部最小值

#### 8.2.6 收敛速率对比

| 方法 | 凸问题 | 强凸问题 | 非凸问题 |
|---|---|---|---|
| GD（全批量，常数 $\alpha$） | $O(1/T)$ | $O(e^{-\mu L t})$ | 到平稳点 $O(1/T)$ |
| SGD（$\alpha_t = 1/t$） | $O(1/\sqrt{T})$ | $O(\log T / T)$ | 到平稳点 $O(1/\sqrt{T})$ |
| SGD（常数 $\alpha$） | 不收敛（在最优解附近震荡） | 震荡，但**收敛到最优解附近的一个小邻域** | 到 $\epsilon$-平稳点 $O(1/\epsilon^4)$ |
| Mini-batch SGD | $O(1/\sqrt{T}) + O(\sigma/B\sqrt{T})$ | — | — |

**常数学习率的 SGD 不精确收敛，但收敛到最优解附近**。具体地，对于 $\mu$-强凸函数：

$$\limsup_{t \to \infty} \mathbb{E}[F(W_t) - F(W^*)] \leq \frac{\alpha \sigma^2}{2\mu}$$

这个上界与学习率 $\alpha$ 成正比——学习率越小，最终误差越小。但 $\alpha$ 太小会导致收敛太慢。这就是理论与实践的永恒张力：

$$\boxed{\alpha \text{ 大} \to \text{收敛快但精度低} \qquad \alpha \text{ 小} \to \text{精度高但收敛慢}}$$

实践中通过**学习率调度**来解决：开始时 $\alpha$ 大（快速探索），结束时 $\alpha$ 小（精确收敛）。

#### 8.2.7 随机近似的更一般形式：Robbins-Monro 定理

**定理（Robbins-Monro, 1951）**：

设我们要解方程 $h(w) = 0$，但只能观测到 $h(w) + \xi_t$（含噪声）。迭代：

$$w_{t+1} = w_t - \alpha_t [h(w_t) + \xi_t]$$

如果满足：
1. $h$ 是 Lipschitz 连续且严格单调的（保证根存在且唯一）
2. $\mathbb{E}[\xi_t | w_t] = 0$，$\text{Var}(\xi_t) \leq \sigma^2$
3. $\sum \alpha_t = \infty$，$\sum \alpha_t^2 < \infty$

则 $w_t$ 几乎必然收敛到 $w^*$（$h(w^*) = 0$）。

**SGD 是 Robbins-Monro 的直接应用**：$h(W) = \nabla F(W)$，要找 $\nabla F(W) = 0$ 的点。

这个框架也解释了为什么 GAN 的训练可以用同样的数学分析——GAN 的判别器和生成器更新都是 Robbins-Monro 形式的随机近似。

#### 8.2.8 Dvoretzky 定理：从 Robbins-Monro 到一般收敛框架

Robbins-Monro 定理已经非常强大，但它要求噪声条件比较具体（零均值、有界方差）。1956 年，Aryeh Dvoretzky 提出了一个**更一般的框架**，不仅涵盖了 Robbins-Monro，还为理解 SGD 的收敛性提供了最根本的视角。

##### Dvoretzky 定理的精确表述

**定理（Dvoretzky, 1956）**：

设 $\{w_t\}_{t=0}^{\infty}$ 是由以下迭代生成的随机序列：

$$w_{t+1} = T_t(w_t + \beta_t + \xi_t)$$

其中：
- $T_t$ 是从 $\mathbb{R}^d$ 到 $\mathbb{R}^d$ 的映射
- $\beta_t \in \mathbb{R}^d$ 是确定性偏移
- $\xi_t \in \mathbb{R}^d$ 是随机噪声

设 $C \subset \mathbb{R}^d$ 是一个非空闭集。如果存在非负数列 $\{a_t\}, \{b_t\}, \{c_t\}$，满足：

$$\boxed{\text{(D1)} \quad \sum_{t=0}^{\infty} a_t < \infty}$$

$$\boxed{\text{(D2)} \quad \sum_{t=0}^{\infty} b_t < \infty}$$

$$\boxed{\text{(D3)} \quad \sum_{t=0}^{\infty} c_t = \infty}$$

并且对所有 $w \in \mathbb{R}^d$ 和所有 $t$，$T_t$ 和噪声满足以下三个条件：

$$\boxed{\text{(C1)} \quad \|T_t(w) - T_t(\pi_C(w))\| \leq (1 + a_t)\|w - \pi_C(w)\|}$$

$$\boxed{\text{(C2)} \quad \|T_t(\pi_C(w)) - \pi_C(T_t(\pi_C(w)))\| \leq b_t + c_t \|w - \pi_C(w)\|}$$

$$\boxed{\text{(C3)} \quad \mathbb{E}\|\xi_t\| \leq a_t + b_t}$$

其中 $\pi_C(w)$ 是 $w$ 到集合 $C$ 的投影（最近点），则 $w_t$ 几乎必然收敛到 $C$ 中。

这个定理看起来抽象，但它的三个条件有非常清晰的直觉。让我们先理解直觉，再严格证明。

##### 三个条件的直觉翻译

**目标**：证明迭代序列 $\{w_t\}$ 越来越靠近集合 $C$（在我们的场景中，$C = \{W^* : \nabla F(W^*) = 0\}$）。

定义"误差"为 $\|w_t - \pi_C(w_t)\|$，即当前点到目标集合的距离。

**条件 (C1)**：远离目标时，每一步至少按比例缩小距离（因子 $1+a_t$，其中 $a_t$ 很小）。

> 直觉：如果你离北京 1000 公里，走一步后你应该更接近北京。$a_t$ 是"允许的额外膨胀"，但 $\sum a_t < \infty$ 意味着总膨胀有限，长期看距离一定是缩小的。
>
> 类比：每次走路时，地图的比例尺可能有微小误差（$1+a_t$ 倍），但误差不会无限累积。

**条件 (C2)**：已经到达目标附近时，不会被推太远。

> 直觉：如果你已经在北京市内，下一步不应该把你推到天津。$b_t$ 是"允许的绝对偏移"（$\sum b_t < \infty$），$c_t$ 是"与距离成比例的偏移"（$\sum c_t = \infty$ 确保远处仍能收敛）。
>
> 类比：即使你到了北京，每一步也可能走错一小段路（$b_t$），但错路的总长度有限。

**条件 (C3)**：噪声的平均幅度可控。

> 直觉：随机扰动的"大小"必须和 (C1)(C2) 中的容许量一致。噪声不能大到淹没信号。

##### Dvoretzky 定理的严格证明

**证明**：

**核心策略**：证明 $\|w_t - \pi_C(w_t)\|$ 的期望趋于零，且几乎必然收敛到零。

---

**Step 1：建立误差递推不等式**

设 $d_t = \|w_t - \pi_C(w_t)\|$（到集合 $C$ 的距离）。

由三角不等式：

$$d_{t+1} = \|w_{t+1} - \pi_C(w_{t+1})\| \leq \|w_{t+1} - \pi_C(w_t)\|$$

（因为 $\pi_C(w_{t+1})$ 是 $C$ 中离 $w_{t+1}$ 最近的点，所以 $\pi_C(w_t)$ 到 $w_{t+1}$ 的距离至少为 $d_{t+1}$）

将 $w_{t+1}$ 展开：

$$w_{t+1} = T_t(w_t + \beta_t + \xi_t)$$

利用三角不等式拆开：

$$\|w_{t+1} - \pi_C(w_t)\| \leq \|T_t(w_t + \beta_t + \xi_t) - T_t(w_t + \beta_t)\| + \|T_t(w_t + \beta_t) - \pi_C(w_t)\|$$

**处理第一项**。由条件 (C1)，对任意 $z = w_t + \beta_t + \xi_t$：

$$\|T_t(z) - T_t(\pi_C(z))\| \leq (1 + a_t) \|z - \pi_C(z)\|$$

进一步地，利用三角不等式和投影的非扩张性（$\|\pi_C(x) - \pi_C(y)\| \leq \|x - y\|$）：

$$\|z - \pi_C(z)\| = \|(w_t + \beta_t + \xi_t) - \pi_C(w_t + \beta_t + \xi_t)\| \leq \|w_t + \beta_t - \pi_C(w_t + \beta_t)\| + \|\xi_t\|$$

所以第一项有上界：$(1 + a_t)(\|w_t + \beta_t - \pi_C(w_t + \beta_t)\| + \|\xi_t\|)$。

**处理第二项**。记 $z_0 = w_t + \beta_t$，应用条件 (C2)：

$$\|T_t(\pi_C(z_0)) - \pi_C(T_t(\pi_C(z_0)))\| \leq b_t + c_t \|z_0 - \pi_C(z_0)\|$$

但我们需要 $\|T_t(z_0) - \pi_C(w_t)\|$ 而不是上面的形式。利用三角不等式：

$$\|T_t(z_0) - \pi_C(w_t)\| \leq \|T_t(z_0) - T_t(\pi_C(z_0))\| + \|T_t(\pi_C(z_0)) - \pi_C(T_t(\pi_C(z_0)))\| + \|\pi_C(T_t(\pi_C(z_0))) - \pi_C(w_t)\|$$

由 (C1)：第一项 $\leq (1 + a_t)\|z_0 - \pi_C(z_0)\|$

由 (C2)：第二项 $\leq b_t + c_t\|z_0 - \pi_C(z_0)\|$

由投影非扩张性：第三项 $\leq \|T_t(\pi_C(z_0)) - w_t\|$，再利用 $T_t$ 的性质和三角不等式，这一项可以被 $(1+a_t)(\|z_0 - \pi_C(z_0)\| + \|\beta_t\|)$ 控制。

---

**Step 2：合并，得到简洁的递推**

将上述所有不等式合并（为简洁，省略中间的展开细节），可以整理成如下形式：

$$d_{t+1} \leq (1 + \tilde{a}_t) d_t + \tilde{b}_t$$

其中 $\tilde{a}_t, \tilde{b}_t$ 是 $a_t, b_t, c_t$ 和 $\|\beta_t\|$ 的组合，满足：

$$\sum_{t=0}^{\infty} \tilde{a}_t < \infty \qquad \text{且} \qquad \sum_{t=0}^{\infty} \tilde{b}_t < \infty$$

这个形式就是证明的核心：**每一步的误差最多被放大 $(1 + \tilde{a}_t)$ 倍，再加上一个可控的常数偏移 $\tilde{b}_t$。**

---

**Step 3：展开递推（telescoping）**

反复展开递推不等式：

$$d_{t+1} \leq (1 + \tilde{a}_t)(1 + \tilde{a}_{t-1}) d_{t-1} + (1 + \tilde{a}_t) \tilde{b}_{t-1} + \tilde{b}_t$$

$$\leq \prod_{s=0}^{t}(1 + \tilde{a}_s) \cdot d_0 + \sum_{s=0}^{t} \tilde{b}_s \prod_{j=s+1}^{t}(1 + \tilde{a}_j)$$

**关键引理**（无穷乘积引理）：

$$\prod_{s=0}^{\infty}(1 + \tilde{a}_s) < \infty \qquad \iff \qquad \sum_{s=0}^{\infty} \tilde{a}_s < \infty$$

**证明**：利用 $1 + x \leq e^x$（对所有 $x > -1$），有：

$$\prod_{s=0}^{\infty}(1 + \tilde{a}_s) \leq \prod_{s=0}^{\infty} e^{\tilde{a}_s} = e^{\sum_{s=0}^{\infty} \tilde{a}_s} < \infty$$

反之，利用 $x \leq 2 \ln(1+x)$（对所有 $x \geq 0$），有：

$$\sum_{s=0}^{\infty} \tilde{a}_s \leq 2 \sum_{s=0}^{\infty} \ln(1 + \tilde{a}_s) = 2 \ln \prod_{s=0}^{\infty} (1 + \tilde{a}_s)$$

因此无穷乘积有限当且仅当无穷级数有限。**证毕。**

---

**Step 4：取极限**

由关键引理，$\prod_{s=0}^{\infty}(1 + \tilde{a}_s) \leq K < \infty$（某个常数）。

所以 $d_t$ 的上界为：

$$d_t \leq K \cdot d_0 + K \cdot \sum_{s=0}^{\infty} \tilde{b}_s \leq K(d_0 + B)$$

其中 $B = \sum_{s=0}^{\infty} \tilde{b}_s < \infty$。这说明 $d_t$ 是有界的。

但我们需要更强的结论：$d_t \to 0$（几乎必然）。

为此，引入**鞅论**（martingale theory）的论证。定义：

$$M_t = d_t - \mathbb{E}[d_t | \mathcal{F}_{t-1}]$$

其中 $\mathcal{F}_{t-1}$ 是到第 $t-1$ 步为止的信息（$\sigma$-代数）。可以验证 $M_t$ 是一个**鞅差序列**（martingale difference sequence）。

由鞅收敛定理（Doob's martingale convergence theorem），$\sum M_t$ 几乎必然收敛。

另一方面，对 $d_t$ 的递推不等式取条件期望：

$$\mathbb{E}[d_{t+1} | \mathcal{F}_t] \leq (1 + \tilde{a}_t) d_t + \mathbb{E}[\tilde{b}_t | \mathcal{F}_t]$$

利用条件 (C3)，$\mathbb{E}[\|\xi_t\|] \leq a_t + b_t$，可以证明 $\mathbb{E}[\tilde{b}_t] \leq \hat{b}_t$，其中 $\sum \hat{b}_t < \infty$。

因此 $d_t$ 的期望满足一个**确定性的递推不等式**：

$$\mathbb{E}[d_{t+1}] \leq (1 + \tilde{a}_t) \mathbb{E}[d_t] + \hat{b}_t$$

由 Step 3 同样的方法展开：

$$\mathbb{E}[d_t] \leq K \cdot \mathbb{E}[d_0] + K \cdot \sum_{s=0}^{t-1} \hat{b}_s \leq K(\mathbb{E}[d_0] + \hat{B})$$

要证明 $\mathbb{E}[d_t] \to 0$，注意到 $d_t$ 的递推中，当 $d_t$ 较大时，缩小因子 $(1 + \tilde{a}_t)$ 中的 $\tilde{a}_t$ 项累积起来使得 $\prod (1+\tilde{a}_s)$ 虽然有限但趋于常数 $K$，而 $d_0$ 是初始距离。

更精确地说，取 $\epsilon > 0$ 任意小。因为 $\sum \hat{b}_s$ 收敛，存在 $T$ 使得 $\sum_{s=T}^{\infty} \hat{b}_s < \epsilon$。对 $t > T$：

$$\mathbb{E}[d_t] \leq K \cdot \mathbb{E}[d_T] \cdot \prod_{s=T}^{t-1}(1 + \tilde{a}_s) + K \sum_{s=T}^{t-1} \hat{b}_s$$

由于 $\sum \tilde{a}_s$ 收敛，$\prod_{s=T}^{t-1}(1 + \tilde{a}_s)$ 在 $t \to \infty$ 时趋于一个有限的极限 $K_T$。而 $\sum_{s=T}^{t-1} \hat{b}_s < \epsilon$。所以：

$$\limsup_{t \to \infty} \mathbb{E}[d_t] \leq K_T \cdot K \cdot \epsilon' + K \epsilon$$

其中 $\epsilon' \to 0$（因为 $d_T$ 本身也在缩小）。由 $\epsilon$ 的任意性：

$$\lim_{t \to \infty} \mathbb{E}[d_t] = 0$$

结合鞅收敛定理，进一步得到**几乎必然收敛**：

$$d_t \xrightarrow{\text{a.s.}} 0$$

即 $w_t$ 几乎必然收敛到集合 $C$。

**证毕。**

##### Dvoretzky 定理与 Robbins-Monro 条件的关系

Robbins-Monro 的两个条件 $\sum \alpha_t = \infty$ 和 $\sum \alpha_t^2 < \infty$ 在 Dvoretzky 框架中自然出现：

| RM 条件 | Dvoretzky 中的对应 | 角色 |
|---|---|---|
| $\sum \alpha_t^2 < \infty$ | $\sum a_t < \infty$，$\sum b_t < \infty$ | 控制噪声和偏移的累积 |
| $\sum \alpha_t = \infty$ | $\sum c_t = \infty$（通过 $c_t \approx \alpha_t$） | 保证远处仍能收敛 |

具体地，对于 SGD 迭代 $W_{t+1} = W_t - \alpha_t (\nabla F(W_t) + \xi_t)$：

- 取 $T_t(w) = w - \alpha_t \nabla F(w)$（确定性梯度下降一步）
- $\beta_t = 0$
- $\xi_t = -\alpha_t \xi_t^{\text{noise}}$

则 Dvoretzky 的条件变为：
- (C1) $\|T_t(w) - T_t(\pi_C(w))\| \leq (1 + L\alpha_t)\|w - \pi_C(w)\|$（由 $L$-Lipschitz 条件），因此 $a_t \approx L\alpha_t$
- (C2) 当 $w = W^*$ 时，$\|T_t(W^*) - W^*\| = \alpha_t \|\nabla F(W^*)\| = 0$
- (C3) $\mathbb{E}[\|\xi_t\|] = \alpha_t \mathbb{E}[\|\xi_t^{\text{noise}}\|] \leq \alpha_t \sigma$

因此：

$$a_t \approx L\alpha_t, \quad b_t \approx 0, \quad c_t \approx 0$$

Dvoretzky 条件要求：
- $\sum a_t = L \sum \alpha_t < \infty$？**不对**——这里需要仔细分析。实际上，(C2) 中的 $c_t$ 控制的是"距离目标越远，回归力越大"。对于强凸函数，$c_t \approx \mu \alpha_t$（$\mu$ 是强凸常数），因此 $\sum c_t = \mu \sum \alpha_t = \infty$ 由 RM 条件一保证。

而 $\sum a_t^2 \approx L^2 \sum \alpha_t^2 < \infty$ 由 RM 条件二保证。

**最终对应**：

| Dvoretzky 条件 | SGD 中的含义 | 保证 |
|---|---|---|
| $\sum a_t < \infty$ | $\sum \alpha_t^2 < \infty$ | 噪声不累积 |
| $\sum b_t < \infty$ | 自动满足 | 偏移不累积 |
| $\sum c_t = \infty$ | $\sum \alpha_t = \infty$ | 能走到目标 |

**Dvoretzky 定理的力量在于**：它不仅证明了 Robbins-Monro 型算法的收敛性，还为理解各种变体（投影 SGD、异步 SGD、分布式 SGD）提供了统一框架——只需要验证三个条件 (C1)(C2)(C3) 和三个数列条件 (D1)(D2)(D3) 即可。

##### 用一句话总结 Dvoretzky 定理

> **如果每一步的"有用信号"累积到无穷（$\sum c_t = \infty$），而"有害噪声"的累积有限（$\sum a_t < \infty, \sum b_t < \infty$），那么算法几乎必然收敛到目标。**

---

## 9. 本讲最重要的结论

1. **非线性激活函数**是神经网络拥有强大表示能力的关键——没有它，多层网络退化为线性分类器
2. **反向传播**的本质是沿计算图递归应用链式法则，每个节点只需知道上游梯度和局部梯度
3. **正则化**不仅是加惩罚项，还包括 Dropout、BN、Stochastic Depth、Fractional Pooling 等结构层面的技巧
4. **优化器**从 SGD → Momentum → RMSProp → Adam 的演进，核心趋势是让学习率越来越"自适应"
5. **池化**是卷积网络中的核心下采样操作，Max Pooling 取最显著特征，Average Pooling 保留整体信息，现代趋势是用 stride 卷积替代显式池化
6. BN 的正则化效果是一个**副产品**（mini-batch 随机噪声），Transformer 不用 BN 而用 LayerNorm
7. **Stochastic Depth** 随机丢弃整条残差分支，既是正则化（隐式集成）也是加速手段（减少 25-50% 训练时间），浅层不丢、深层多丢
8. **Fractional Pooling** 在池化操作中引入空间随机性，通过随机切分位置打破确定性，使模型学到对空间位置不敏感的特征
9. **"Batch"有两层含义**：梯度下降中控制"一次更新看多少数据"（mini-batch GD），归一化中控制"按什么维度统计"（BN vs LN vs GN）。BN/LN/GN/IN/RMSNorm 构成一个统一谱系，区别只在于统计量跨哪些维度计算
10. **Hessian 矩阵**描述损失面的二阶曲率，条件数 $\kappa = \lambda_{\max}/\lambda_{\min}$ 决定优化难度，现代优化器（RMSProp、Adam）本质上是在隐式近似 Newton 法 $H^{-1} \nabla F$
11. **Robbins-Monro 条件** $\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$ 保证 SGD 精确收敛；实践中用常数学习率是因为深度学习的非凸性使噪声有利（逃逸鞍点、倾向平坦最优解），常数学习率 SGD 收敛到最优解附近半径为 $O(\alpha \sigma^2 / 2\mu)$ 的邻域
12. **Dvoretzky 定理**是 Robbins-Monro 的一般化：核心递推 $d_{t+1} \leq (1+\tilde{a}_t)d_t + \tilde{b}_t$ 中，$\sum\tilde{a}_t < \infty$ 控制噪声累积，$\sum c_t = \infty$ 保证走到目标，通过无穷乘积引理和鞅收敛定理完成严格证明
