# 第7讲：深度学习（一）—— 神经网络基础知识

> 丁恒辉，2026/4/16 | 基于 PDF 课件 + 课堂转录 + 补充推导整理

---

## 本讲概览

本讲是深度学习的入门课，围绕三条主线展开：

1. **神经网络**：从线性分类器到多层非线性网络
2. **反向传播**：计算图 + 链式法则高效求梯度
3. **正则化与优化**：防止过拟合，找到更好的解

---

## 知识地图

```
神经网络基础
├── 从线性分类器 → 两层网络 → 深层网络
│   └── 非线性激活函数（ReLU 最好）
├── 反向传播
│   ├── 计算图 + 链式法则
│   ├── 门电路规则（加法、乘法、复制、Max）
│   └── PyTorch 中的 forward / backward
├── 正则化
│   ├── L1 / L2 / Elastic Net
│   ├── Dropout（随机丢弃神经元）
│   ├── Batch Normalization（副产品正则化）
│   ├── Stochastic Depth / DropPath（随机丢弃残差分支）
│   └── Fractional Pooling（随机池化切分）
├── 优化方法
│   ├── SGD → SGD+Momentum → RMSProp → Adam
│   ├── 学习率调度（step / cosine / warmup / inverse sqrt）
│   └── Batch Size 与学习率的关系
├── 数学深入
│   ├── Hessian 矩阵（二阶曲率与条件数）
│   ├── Robbins-Monro 条件（∑α=∞, ∑α²<∞）
│   └── Dvoretzky 定理（一般收敛框架）
└── 池化（Pooling）
    ├── Max / Average / Global Average
    └── 现代 trend：stride 卷积替代显式池化
```

---

## 核心公式速查

| 主题 | 公式 |
|---|---|
| 两层网络 | $s = W_2 \sigma(W_1 x + b_1) + b_2$ |
| 链式法则 | $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q} \cdot \frac{\partial q}{\partial x}$ |
| 全损失 | $L = L_{\text{data}} + \lambda R(W)$ |
| SGD+Momentum | $v_t = \rho v_{t-1} + \nabla L;\; W \leftarrow W - \alpha v_t$ |
| RMSProp | $v_t = \rho v_{t-1} + (1-\rho)(\nabla L)^2;\; W \leftarrow W - \frac{\alpha}{\sqrt{v_t}+\epsilon}\nabla L$ |
| Adam | Momentum + RMSProp 的结合，加偏差校正 |
| Batch Norm | $\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}};\; y = \gamma\hat{x} + \beta$ |
| Hessian 条件数 | $\kappa = \lambda_{\max} / \lambda_{\min}$ |
| RM 条件 | $\sum\alpha_t = \infty;\; \sum\alpha_t^2 < \infty$ |

---

## 正则化方法统一视角

所有正则化方法共享一个哲学：**用随机性打破确定性计算路径**。

| 方法 | 随机化对象 | 粒度 |
|---|---|---|
| Dropout | 单个神经元 | 细（神经元级） |
| Stochastic Depth | 整条残差分支 | 粗（层/块级） |
| Fractional Pooling | 空间池化切分位置 | 结构（空间级） |
| Batch Normalization | mini-batch 统计量 | 批次级 |

---

## 归一化方法统一谱系

所有归一化方法都在做"让数据保持好范围"，区别仅在于**按什么维度统计**：

| 方法 | 统计维度 | 依赖 batch | 主要场景 |
|---|---|---|---|
| Batch Norm | (N, H, W) per channel | 是 | CNN |
| Layer Norm | (C, H, W) per sample | 否 | Transformer |
| Group Norm | (C/G, H, W) per group | 否 | 小 batch CV |
| Instance Norm | (H, W) per (sample, channel) | 否 | 风格迁移 |
| RMS Norm | 同 LN，只算方差 | 否 | LLaMA 等大模型 |

> $G=1$ 时 Group Norm = Layer Norm；$G=C$ 时 = Instance Norm。

---

## Batch Size 与学习率

**线性缩放规则**：$B$ 增大 $k$ 倍 → $\alpha$ 增大 $k$ 倍（$B \leq 512$ 时有效）

本质推导：
- 每 epoch 步数 $= N/B$，总信号 $= (N/B) \cdot \alpha$，令其不变得 $\alpha \propto B$
- 每 epoch 总噪声方差 $= N\alpha^2\sigma^2/B^2$，代入 $\alpha = cB$ 得 $= Nc^2\sigma^2$（也与 $B$ 无关）

线性缩放保持"每个 epoch 的总推动量和总噪声方差不变"，但噪声频率降低导致泛化变差。

| Batch Size | 推荐学习率 | 策略 |
|---|---|---|
| 32 | 0.1 | 基准 |
| 64~256 | 0.2~0.8 | 线性缩放 |
| 512~1024 | 1.0~1.5 | 线性缩放 + warmup |
| 2048+ | 1.0~2.0 | 平方根缩放 + warmup + 可能 LARS |

---

## 关键定理

**Robbins-Monro (1951)**：$\sum\alpha_t=\infty$ 保证能走到最优解；$\sum\alpha_t^2<\infty$ 保证噪声不累积。$\alpha_t = 1/t$ 恰好同时满足。

**Dvoretzky (1956)**：RM 的一般化。核心递推 $d_{t+1} \leq (1+\tilde{a}_t)d_t + \tilde{b}_t$，通过无穷乘积引理（$\prod(1+a_s) < \infty \iff \sum a_s < \infty$）和鞅收敛定理证明几乎必然收敛。

**实践中用常数学习率**：因为深度学习的非凸性使噪声有益（逃逸鞍点、倾向平坦最优解）。常数 $\alpha$ 的 SGD 在强凸情况下收敛到最优解附近半径为 $O(\alpha\sigma^2/2\mu)$ 的邻域。

---

## 课堂原话摘录

> "不要因为大模型容易过拟合就刻意去选参数量小的模型，可以用正则化的方法来避免过拟合。"
>
> "我们升维度，永远都是为了……"（将非线性不可分的数据变换到可分空间）
>
> "如果没有非线性激活函数会怎样？我们还是得到了一个线性分类器。"

---

## 文件索引

| 文件 | 内容 |
|---|---|
| `lec7.md` | 完整笔记，含数学推导、伪代码、证明、深度分析 |
| `人工智能-7-深度学习（一）.pdf` | 145 页课件原文 |
| `transcript.md` | 课堂转录 |
