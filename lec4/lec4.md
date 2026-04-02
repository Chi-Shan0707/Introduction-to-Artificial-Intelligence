# 第4讲：损失函数、梯度下降与线性模型

## 本讲提要

- 回顾：分类任务与学习目标
- 相关主题：密度估计、降维、聚类
- 损失函数体系：回归、分类、分布匹配与度量学习
- 优化方法：梯度下降、SGD、Mini-batch 与常见改进
- 线性模型：线性回归、逻辑回归、Hinge、感知机

---

## 损失函数

损失函数（Loss Function）是机器学习里非常核心的概念：

- 模型在做预测后，损失函数用一个数值衡量“预测得有多差”
- 训练的目标就是让这个损失尽可能小
- 优化器（如梯度下降）沿着“让损失变小”的方向更新参数

可以把它理解为：

- 预测值和真实值差得越多，损失越大
- 损失越大，模型“挨罚”越重

---

## 1. 损失函数、目标函数、代价函数的关系

- 单样本损失：一个样本的误差
- 数据集平均损失（经验风险）：所有样本损失的平均
- 目标函数：训练时真正优化的函数，通常是

$$
\text{Objective} = \frac{1}{N}\sum_{i=1}^{N} \ell\left(y_i, \hat y_i\right) + \lambda \Omega(\theta)
$$

其中：

- $\ell$ 是损失函数
- $\Omega(\theta)$ 是正则项（控制模型复杂度）
- $\lambda$ 是正则强度

---

## 2. 回归任务常见损失函数

回归任务：预测连续值，例如房价、温度、销量。

### 2.1 MSE（均方误差）

公式：

$$
\ell_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat y_i)^2
$$

特点：

- 平滑、可导，优化方便
- 对大误差惩罚很重（平方）
- 对离群点敏感

适用场景：

- 误差接近高斯分布
- 希望强力压制大误差

房价预测小例子：

- 真实房价（万元）：$[100, 120, 150]$
- 预测房价（万元）：$[110, 115, 170]$
- 误差：$[-10, 5, -20]$
- 平方误差：$[100, 25, 400]$
- MSE：$\frac{100 + 25 + 400}{3} = 175$

可以看到第三个样本误差 20，被平方后变成 400，对整体影响特别大。

### 2.2 RMSE（均方根误差）

公式：

$$
\ell_{\text{RMSE}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat y_i)^2}
$$

特点：

- 与原始量纲一致（比如“万元”）
- 比 MSE 更直观

适用场景：

- 需要给业务方解释误差大小时常用

### 2.3 MAE（平均绝对误差）

公式：

$$
\ell_{\text{MAE}} = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat y_i|
$$

特点：

- 对离群点更鲁棒（不像 MSE 会平方放大）
- 梯度在 0 点不平滑（但实践中可处理）

适用场景：

- 数据里离群点较多
- 希望“平均偏差”更稳健

房价预测同样例：

- 绝对误差：$[10, 5, 20]$
- MAE：$\frac{10 + 5 + 20}{3} = 11.67$

### 2.4 Huber Loss（平滑 L1）

定义（误差 $e = y - \hat y$）：

$$
\ell_\delta(e)=
\begin{cases}
\frac{1}{2}e^2, & |e|\le \delta \\
\delta(|e| - \frac{1}{2}\delta), & |e|>\delta
\end{cases}
$$

特点：

- 小误差时像 MSE（平滑）
- 大误差时像 MAE（抗离群点）
- 折中效果好

适用场景：

- 既想保留 MSE 的可优化性，又担心离群点

### 2.5 Log-Cosh Loss

公式：

$$
\ell = \sum_i \log\left(\cosh(\hat y_i - y_i)\right)
$$

特点：

- 小误差近似平方误差
- 大误差近似绝对误差
- 比 Huber 更平滑

适用场景：

- 需要非常平滑梯度的回归问题

---

## 3. 二分类常见损失函数

二分类：例如苹果 vs 香蕉（apple vs banana）。

设：

- $y \in \{0,1\}$
- $\hat p = P(y=1|x)$ 是模型预测为正类的概率

### 3.1 0-1 Loss（理论常见，实践少直接优化）

公式：

$$
\ell_{0-1}(y,\hat y)=
\begin{cases}
0, & y=\hat y \\
1, & y\ne\hat y
\end{cases}
$$

特点：

- 非常直观：错一个就罚 1
- 不可导，难以直接用梯度法优化

### 3.2 BCE / Log Loss（二元交叉熵，最经典）

公式：

$$
\ell_{\text{BCE}} = -\left[y\log(\hat p) + (1-y)\log(1-\hat p)\right]
$$

特点：

- 可导，适合梯度优化
- 对“自信但错误”的预测惩罚很重
- 输出概率，有良好概率解释

适用场景：

- 绝大多数二分类神经网络/逻辑回归

apple vs banana 小例子：

1) 样本 A，真实是苹果（$y=1$）

- 模型给 $\hat p=0.9$（很自信且正确）
- 损失：$-\log(0.9) \approx 0.105$

2) 样本 B，真实是苹果（$y=1$）

- 模型给 $\hat p=0.1$（很自信但错）
- 损失：$-\log(0.1) \approx 2.303$

说明：错误且自信时，损失暴涨，这能强烈推动模型修正。

### 3.3 Hinge Loss（SVM 经典）

先把标签写成 $y\in\{-1,+1\}$，模型输出实数分数 $f(x)$。

公式：

$$
\ell_{\text{hinge}} = \max(0, 1 - y f(x))
$$

特点：

- 强调“间隔（margin）”
- 不仅要分对，还要分得“够开”

适用场景：

- SVM 或关注分类间隔的问题

简单例子：

- 若 $y=+1$，$f(x)=2$，则损失 $\max(0,1-2)=0$（分对且间隔够）
- 若 $y=+1$，$f(x)=0.2$，则损失 $0.8$（分对但不够自信）

### 3.4 Focal Loss（处理类别不平衡）

二分类常见形式：

$$
\ell_{\text{focal}} = -\alpha(1-p_t)^\gamma\log(p_t)
$$

其中：

- 若样本分类正确且 $p_t$ 高，$(1-p_t)^\gamma$ 很小，损失被压低
- 模型会更关注“难样本”

适用场景：

- 正负样本极不平衡（如目标检测、欺诈检测）

---

## 4. 多分类常见损失函数

例如：猫/狗/鸟 三分类。

### 4.1 Softmax Cross-Entropy（多分类交叉熵）

模型输出 logits：$z_1,\dots,z_K$，Softmax 概率：

$$
\hat p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

若真实类别是 $c$，损失：

$$
\ell = -\log(\hat p_c)
$$

特点：

- 最常用多分类损失
- 数值稳定版本在框架里一般已封装好

适用场景：

- 图像分类、文本分类等标准多分类问题

### 4.2 Label Smoothing Cross-Entropy

思想：不把真实标签设为绝对 1/0，而是“稍微平滑”。

- 原 one-hot：真实类 1，其它类 0
- 平滑后：真实类 $1-\varepsilon$，其它类分到 $\varepsilon$

作用：

- 减少过度自信
- 提升泛化，缓解过拟合

适用场景：

- 大规模分类模型、Transformer 训练中很常见

---

## 5. 概率分布匹配类损失

### 5.1 KL Divergence（KL 散度）

离散形式：

$$
D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

理解：

- 用分布 $Q$ 近似真实分布 $P$ 时的信息损失
- 非对称：$D_{KL}(P\|Q) \ne D_{KL}(Q\|P)$

适用场景：

- 知识蒸馏
- 变分推断（如 VAE）
- 分布对齐任务

### 5.2 JS Divergence（Jensen-Shannon）

基于 KL 的对称版本，更稳定，常见于 GAN 理论分析。

---

## 6. 排序、检索和度量学习常见损失

### 6.1 Contrastive Loss

用于“相似/不相似”样本对。希望：

- 相似样本在嵌入空间距离更近
- 不相似样本距离超过间隔

常见于人脸验证、图像检索。

### 6.2 Triplet Loss

三元组：anchor、positive、negative。

目标：

$$
d(a,p) + m < d(a,n)
$$

损失通常写作：

$$
\ell = \max\{0, d(a,p)-d(a,n)+m\}
$$

适用场景：

- 人脸识别（同人更近，不同人更远）

---

## 7. 序列任务常见损失

### 7.1 CTC Loss

适合输入输出对齐未知的序列任务，如语音识别、OCR。

核心价值：

- 不需要逐帧标注对齐
- 直接学习序列映射

### 7.2 序列交叉熵（Teacher Forcing）

在机器翻译、文本生成中，逐时间步计算交叉熵并求和或平均。

---

## 8. 从任务角度如何选损失函数

### 8.1 回归

- 首选：MSE（简单、经典）
- 离群点多：MAE / Huber
- 解释性（单位直观）：RMSE

### 8.2 二分类

- 首选：BCE
- 间隔导向：Hinge
- 类别不平衡：Focal Loss（可配合重采样/类别权重）

### 8.3 多分类

- 首选：Softmax Cross-Entropy
- 过拟合明显、过度自信：加 Label Smoothing

### 8.4 分布学习

- KL、JS 视具体模型目标使用

---

## 9. 统一小例子串起来理解

用两个场景贯通：

1) 苹果/香蕉分类（离散标签）
2) 房价预测（连续值）

### 9.1 苹果/香蕉分类

- 输入：颜色、纹理、形状等特征
- 输出：是苹果的概率 $\hat p$
- 损失：BCE

当真实是苹果：

- 预测 0.95，损失很小
- 预测 0.05，损失很大

训练过程会不断把“正确类别概率”往上推。

### 9.2 房价预测

- 输入：面积、地段、楼龄
- 输出：连续值 $\hat y$（预测价格）
- 损失：MSE 或 MAE 或 Huber

如果某个样本误差特别大：

- MSE 会非常在意它（因为平方）
- MAE 相对稳健
- Huber 折中

---

## 10. 常见误区

1. 只看准确率，不看损失

- 准确率可能提升了，但概率校准很差
- 损失更能反映“信心是否合理”

2. 损失下降就一定泛化更好

- 训练损失下降可能伴随过拟合
- 要同时看验证集损失

3. 忽视类别不平衡

- 在不平衡数据上，普通 BCE 可能偏向多数类
- 需考虑 Focal Loss、类别权重、采样策略

4. 不匹配输出层与损失函数

- 二分类通常 sigmoid + BCE
- 多分类通常 softmax + cross-entropy
- 框架里有的损失函数已内置 logits 处理，避免重复激活

---

## 11. 一句话总结每个经典损失

- MSE：大错重罚，回归经典
- MAE：抗离群点，回归稳健
- Huber：MSE 与 MAE 折中
- BCE：二分类概率学习主力
- Hinge：强调分类间隔，SVM 经典
- Cross-Entropy：多分类标准答案
- Focal：不平衡数据的难样本聚焦
- KL：比较两个分布差异
- Triplet/Contrastive：学“相似性距离”
- CTC：序列无对齐训练利器

---

## 12. 贯通总结（重点）

损失函数本质上回答一个问题：

“在当前任务下，什么样的错误最不能接受？”

- 如果你不能接受大偏差：用 MSE
- 如果你担心离群点污染：用 MAE/Huber
- 如果你做分类且要概率解释：用交叉熵
- 如果样本严重不平衡：在交叉熵上加 Focal 思想
- 如果你想学表示空间结构：用对比/三元组损失

所以，选损失函数不是背公式，而是把业务目标翻译成“惩罚规则”。

当你能把“我要惩罚什么错误”说清楚，损失函数就基本选对了。

---

## 13. 梯度下降（Gradient Descent）循序渐进讲解

下面这部分把“损失函数怎么被真正优化”讲清楚。

如果说损失函数定义了“错多少要罚多少”，那梯度下降就是“怎样系统地把罚分降下来”。

### 13.1 先明确目标：我们到底在最小化什么

设模型参数为 $\theta$，目标函数为：

$$
J(\theta) = \frac{1}{N}\sum_{i=1}^{N}\ell\big(y_i,\hat y_i(\theta)\big)
$$

训练本质上是在解：

$$
\theta^* = \arg\min_{\theta} J(\theta)
$$

这句话的意思是：找一组参数，让总体损失最小。

### 13.2 为什么“导数/梯度”能告诉我们怎么走

一元函数里，导数 $f'(x)$ 描述函数在 $x$ 点的变化趋势：

- $f'(x)>0$：往右走函数上升
- $f'(x)<0$：往右走函数下降

想让函数下降，就应沿“负导数方向”走。

多元函数里，导数推广为梯度向量：

$$
\nabla J(\theta)=\left[\frac{\partial J}{\partial \theta_1},\frac{\partial J}{\partial \theta_2},\dots,\frac{\partial J}{\partial \theta_d}\right]^\top
$$

结论：

- 梯度方向是函数增长最快方向
- 所以取负梯度方向 $-\nabla J(\theta)$，就是下降最快方向（局部意义）

### 13.3 梯度下降的核心更新公式

第 $t$ 轮参数记为 $\theta_t$，则更新为：

$$
\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t)
$$

其中：

- $\eta$（eta）是学习率（步长）
- 学习率控制每一步走多远

这就是机器学习里最经典的一条公式。

### 13.4 一维可手算例子（严谨版）

考虑目标函数：

$$
J(w)=(w-3)^2
$$

它的最小值显然在 $w=3$。

1) 求导：

$$
\frac{dJ}{dw}=2(w-3)
$$

2) 设初值 $w_0=0$，学习率 $\eta=0.1$。

3) 迭代：

- 第 0 步：

$$
g_0=2(0-3)=-6,
\quad
w_1=w_0-0.1\cdot g_0=0+0.6=0.6
$$

- 第 1 步：

$$
g_1=2(0.6-3)=-4.8,
\quad
w_2=0.6-0.1\cdot(-4.8)=1.08
$$

- 第 2 步：

$$
g_2=2(1.08-3)=-3.84,
\quad
w_3=1.464
$$

你会看到 $w$ 从 0 不断逼近 3，且越接近最优点，梯度绝对值越小，步子也自然变小。

### 13.5 二维直觉（不靠比喻，靠数学）

若参数是二维向量 $\theta=[w,b]^\top$，那么每一步都要同时更新：

$$
\begin{aligned}
w_{t+1} &= w_t - \eta\frac{\partial J}{\partial w} \\
b_{t+1} &= b_t - \eta\frac{\partial J}{\partial b}
\end{aligned}
$$

这就是线性回归里常见的参数更新形式。

### 13.6 用在线性回归上：从损失到更新

模型：

$$
\hat y_i = w^T\phi(x_i) + b
$$

其中 $\phi(x)$ 是特征映射（可理解为把原始输入变成模型使用的特征向量）。

MSE 损失：

$$
J(w,b)=\frac{1}{N}\sum_{i=1}^N(\hat y_i-y_i)^2
$$

其梯度（可由链式法则推导）：

$$
\frac{\partial J}{\partial w}=\frac{2}{N}\sum_{i=1}^N(\hat y_i-y_i)\phi(x_i),
\qquad
\frac{\partial J}{\partial b}=\frac{2}{N}\sum_{i=1}^N(\hat y_i-y_i)
$$

更新：

$$
w\leftarrow w-\eta\frac{\partial J}{\partial w},
\qquad
b\leftarrow b-\eta\frac{\partial J}{\partial b}
$$

这就是房价预测这类回归任务训练时最基础的数学骨架。

### 13.7 学习率为什么关键

若把学习率写成 $\eta$：

- $\eta$ 太小：每步很短，收敛慢
- $\eta$ 太大：可能越过最优点，甚至震荡或发散

实践上常见策略：

- 先用较大学习率快速下降
- 再逐步衰减（learning rate decay）精细收敛

### 13.8 三种常见梯度下降形式

设训练集有 $N$ 个样本。

1) 批量梯度下降（Batch GD）

- 每轮用全部样本算一次梯度
- 方向稳定，但每次计算成本高

2) 随机梯度下降（SGD）

- 每次用 1 个样本估计梯度
- 速度快、噪声大，轨迹抖动明显

3) 小批量梯度下降（Mini-batch GD）

- 每次用一小批（如 32、64、128）样本
- 兼顾稳定与效率，深度学习里最常用

### 13.9 收敛判据：什么时候停

常见停止条件：

- 达到最大迭代轮数
- 相邻两轮损失下降小于阈值
- 梯度范数足够小：$\|\nabla J(\theta)\| < \epsilon$
- 验证集长期不再提升（早停）

### 13.10 局部最小、鞍点与非凸问题

在深度学习中，$J(\theta)$ 常是非凸函数。

可能遇到：

- 局部最小值
- 鞍点（梯度接近 0 但不是最小值）

实际经验表明：

- 合理初始化 + Mini-batch 噪声 + 学习率调度，通常可找到可用解
- 不必追求“全局最优”才算成功

### 13.11 常见改进算法（本质仍是梯度下降）

1) Momentum（动量）

- 对历史梯度做指数加权平均
- 减少来回震荡，加速沿主方向前进

2) RMSProp

- 对不同参数使用自适应步长
- 在陡峭/平坦方向上更平衡

3) Adam

- 结合 Momentum 与 RMSProp 思想
- 工程里非常常用，默认起点常设为 Adam

### 13.12 与前面损失函数章节贯通

把你已经学过的内容串起来：

- 损失函数定义“怎么罚”
- 梯度给出“往哪改参数”
- 学习率决定“每次改多少”
- 迭代更新实现“持续降低损失”

即：

$$
\text{Define }J(\theta)\ \rightarrow\ \text{Compute }\nabla J(\theta)\ \rightarrow\ \text{Update }\theta\ \rightarrow\ \text{Repeat}
$$

这就是从“建模”到“训练”的主线。

### 13.13 最简流程清单（可直接背）

1. 选模型并定义损失 $J(\theta)$
2. 初始化参数 $\theta_0$
3. 计算梯度 $\nabla J(\theta_t)$
4. 用 $\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t)$ 更新
5. 检查停止条件，不满足则返回第 3 步

如果你把这五步真正理解，梯度下降就不是“公式记忆”，而是可推导、可实现、可调参的方法体系。

### 13.14 NumPy 代码：线性回归中的梯度下降

下面代码用 NumPy 手写线性回归训练，支持三种模式：

- `batch`：批量梯度下降（每次用全部样本）
- `sgd`：随机梯度下降（每次用 1 个样本）
- `mini-batch`：小批量梯度下降（每次用一小批样本）

```python
import numpy as np


def make_toy_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=n)
    noise = rng.normal(0, 1.0, size=n)
    y = 2.0 * x + 1.5 + noise
    return x, y


def mse_and_grads(w, b, x_batch, y_batch):
    # 预测与误差
    y_hat = w * x_batch + b
    err = y_hat - y_batch

    # MSE
    loss = np.mean(err ** 2)

    # 对 w, b 的梯度
    dw = 2.0 * np.mean(err * x_batch)
    db = 2.0 * np.mean(err)
    return loss, dw, db


def train_linear_gd(
    x,
    y,
    mode="batch",
    lr=0.01,
    epochs=200,
    batch_size=32,
    seed=0,
):
    rng = np.random.default_rng(seed)
    n = x.shape[0]

    # 参数初始化
    w, b = 0.0, 0.0
    history = []

    for epoch in range(epochs):
        if mode == "batch":
            loss, dw, db = mse_and_grads(w, b, x, y)
            w -= lr * dw
            b -= lr * db

        elif mode == "sgd":
            # 每次只看 1 个样本，一轮会更新 n 次
            idx = rng.permutation(n)
            for i in idx:
                xi = x[i:i + 1]
                yi = y[i:i + 1]
                _, dw, db = mse_and_grads(w, b, xi, yi)
                w -= lr * dw
                b -= lr * db

            # 记录整数据集上的损失，便于比较
            loss, _, _ = mse_and_grads(w, b, x, y)

        elif mode == "mini-batch":
            # 每次看 batch_size 个样本，一轮会更新 n / batch_size 次
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                xb = x[batch_idx]
                yb = y[batch_idx]
                _, dw, db = mse_and_grads(w, b, xb, yb)
                w -= lr * dw
                b -= lr * db

            loss, _, _ = mse_and_grads(w, b, x, y)

        else:
            raise ValueError("mode must be one of: batch, sgd, mini-batch")

        history.append(loss)

    return w, b, history


if __name__ == "__main__":
    x, y = make_toy_data(n=500, seed=123)

    for mode in ["batch", "sgd", "mini-batch"]:
        w, b, hist = train_linear_gd(
            x,
            y,
            mode=mode,
            lr=0.01,
            epochs=200,
            batch_size=32,
            seed=123,
        )
        print(f"mode={mode:10s}  w={w:.4f}  b={b:.4f}  final_loss={hist[-1]:.4f}")
```

你会看到三种模式都能学到接近真实值的参数（约 $w\approx 2, b\approx 1.5$），但训练轨迹不同。

### 13.15 随机梯度下降（SGD）讲解

SGD 的更新粒度是“单样本”：

- 每次只用 1 个样本估计梯度
- 每处理 1 个样本就更新一次参数

形式上可写为：

$$
\theta \leftarrow \theta - \eta\nabla \ell_i(\theta)
$$

其中 $\ell_i$ 是第 $i$ 个样本的损失。

特点：

- 优点：每步计算非常快，能更早开始更新
- 缺点：梯度噪声大，损失曲线抖动明显，收敛轨迹不稳定
- 工程意义：在大数据场景下常作为基础方案

关于你写的“收敛慢”：

- 按“单步计算成本”看，SGD 很快
- 按“损失曲线平滑收敛到很稳定点”看，SGD 常因噪声而显得慢

这两个“快/慢”不矛盾，它们衡量标准不同。

### 13.16 Mini-batch 讲解

Mini-batch 是 Batch GD 和 SGD 的折中：

- 每次用 $B$ 个样本（如 32、64、128）估计梯度
- 每个 epoch 更新约 $N/B$ 次

更新式：

$$
\theta \leftarrow \theta - \eta\nabla J_{\mathcal{B}}(\theta)
$$

其中 $\mathcal{B}$ 表示当前小批量。

特点：

- 比 SGD 稳定（方差更小）
- 比 Batch GD 高效（每次不必扫全量数据）
- 更适合 GPU/并行计算，因此深度学习中最常用

### 13.17 分组损失（Group Loss）说明

在你当前这部分语境里，“分组损失”通常是指：

- 把训练集分成一个个小组（mini-batch）
- 每次先算这个小组上的平均损失，再做参数更新

若当前小组是 $\mathcal{B}$，大小为 $|\mathcal{B}|=B$，分组损失定义为：

$$
J_{\mathcal{B}}(\theta) = \frac{1}{B}\sum_{i\in\mathcal{B}}\ell_i(\theta)
$$

对应梯度：

$$
\nabla J_{\mathcal{B}}(\theta)=\frac{1}{B}\sum_{i\in\mathcal{B}}\nabla \ell_i(\theta)
$$

参数更新：

$$
\theta \leftarrow \theta - \eta\nabla J_{\mathcal{B}}(\theta)
$$

它和“全数据损失”的关系：

- 全数据损失：$J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta)$
- 分组损失：$J_{\mathcal{B}}(\theta)$ 是 $J(\theta)$ 的随机近似
- 随机抽 batch 时，$\mathbb{E}[\nabla J_{\mathcal{B}}(\theta)]\approx \nabla J(\theta)$（无偏估计）

这就是 mini-batch 能高效训练的数学基础。

另外，“分组损失”有时也指“按类别/人群分组后加权”，常见于类别不平衡或公平性任务：

$$
J(\theta)=\sum_{g=1}^{G}\alpha_g J_g(\theta), \quad \sum_g \alpha_g=1
$$

这里 $g$ 表示组别（如类别组、人群组），$\alpha_g$ 是组权重。

### 13.18 三者对比（考试/复习高频）

- Batch GD：方向最稳，但每步最慢
- SGD：每步最快，但抖动最大
- Mini-batch：通常是实践首选，兼顾速度和稳定性

### 13.19 实践建议

1. 默认先用 Mini-batch（如 batch size = 32 或 64）
2. 学习率先试 $10^{-2}$ 或 $10^{-3}$，再根据曲线调整
3. 固定随机种子（如 NumPy 的 `default_rng(seed)`）提高可复现性
4. 观察训练/验证损失，不只看训练损失

---

## 14. 线性模型专题（结合本讲）

这一节把“线性模型”系统补齐，并和你前面的损失函数、梯度下降串起来。

### 14.1 统一表达：线性打分函数

大多数线性模型都可以写成：

- 打分函数：$f(x)=w^T\phi(x)+b$
- 其中 $x$ 是原始输入，$\phi(x)$ 是特征映射后的向量，$w$ 是参数向量，$b$ 是偏置项

差别主要来自两点：

- 输出层怎么定义（直接输出数值，还是过 sigmoid/softmax）
- 损失函数怎么选（MSE、BCE、交叉熵、hinge 等）

### 14.2 线性回归（Linear Regression）

目标：预测连续值，例如房价。

- 模型：$\hat y = w^T\phi(x)+b$
- 常见损失：MSE

MSE 目标函数（加上可选正则）：

$$
J(w,b)=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2 + \lambda\Omega(w)
$$

其中：

- 若 $\Omega(w)=\|w\|_2^2$，对应 Ridge
- 若 $\Omega(w)=\|w\|_1$，对应 Lasso

训练方法：

- 小数据可用闭式解（正规方程）
- 大数据通常用梯度下降或其变体

正规方程（不加正则时）：

$$
w^*=(X^TX)^{-1}X^Ty
$$

注意：当 $X^TX$ 不可逆时，要用伪逆或加正则。

### 14.3 逻辑回归（Logistic Regression）

虽然名字有“回归”，它是经典二分类模型。

步骤：

1. 先做线性打分：$z=w^T\phi(x)+b$
2. 再做 sigmoid 映射：$p=\sigma(z)=1/(1+e^{-z})$
3. 把 $p$ 解释为“属于正类的概率”

损失函数一般用 BCE：

$$
\ell = -[y\log p + (1-y)\log(1-p)]
$$

决策规则通常是：

- $p\ge 0.5$ 判为正类
- $p<0.5$ 判为负类

几何上，决策边界由 $w^T\phi(x)+b=0$ 给出，是一个超平面。

### 14.4 Softmax 回归（多分类线性模型）

当类别有 $K$ 个时，可对每个类给一个线性打分：

- $z_k = w_k^T\phi(x) + b_k$

再做 softmax 得到概率：

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

损失用多分类交叉熵（你前面已经写过）。

这等价于“线性模型 + softmax 输出层 + 交叉熵损失”。

### 14.5 线性可分与不可分

二分类中，若存在某个超平面能把两类完全分开，叫线性可分。

- 线性可分：线性模型往往表现很好
- 线性不可分：需要特征工程或非线性映射

常见做法：

- 增加多项式特征（如 $x_1^2, x_1x_2$）
- 核方法（如核 SVM）
- 直接使用非线性模型（树、神经网络）

### 14.6 正则化：控制复杂度，提升泛化

#### L2 正则（Ridge）

- 目标：抑制参数过大，降低过拟合
- 形式：$\lambda\|w\|_2^2$
- 特点：参数会变小，但通常不会变成 0

#### L1 正则（Lasso）

- 形式：$\lambda\|w\|_1$
- 特点：会把一部分参数压到 0，具有特征选择效果

#### Elastic Net

- 同时使用 L1 和 L2
- 在高维、强相关特征下常较稳健

### 14.7 概率视角（为什么这些损失合理）

线性模型常可由“最大似然”推出：

- 线性回归 + MSE：对应噪声服从高斯分布
- 逻辑回归 + BCE：对应伯努利分布似然

所以“损失函数”不是随便拍脑袋选，而是统计假设下的自然结果。

### 14.8 与梯度下降的统一模板

从工程角度，训练线性模型几乎都可归纳成：

1. 写出打分函数 $f(x)$
2. 定义输出映射（恒等/sigmoid/softmax）
3. 选损失函数（MSE/BCE/交叉熵）
4. 选优化器（Batch/SGD/Mini-batch/Adam）
5. 加不加正则（L1/L2）

这就是你整章最核心的“统一建模流程”。

### 14.9 小例子串讲

#### 例 1：房价预测

- 模型：线性回归
- 输出：连续值
- 损失：MSE
- 优化：Mini-batch GD
- 可选：L2 正则稳定参数

#### 例 2：apple vs banana

- 模型：逻辑回归
- 输出：正类概率
- 损失：BCE
- 优化：SGD 或 Mini-batch
- 类别不平衡时：可加类别权重或 focal 思路

#### 例 3：猫/狗/鸟三分类

- 模型：softmax 回归
- 输出：三类概率
- 损失：多分类交叉熵

### 14.10 复习抓手（建议背）

- 线性模型的骨架：$f(x)=w^T\phi(x)+b$
- 回归看 MSE，二分类看 BCE，多分类看交叉熵
- 线性模型不等于“简单”，关键在特征表示与正则化
- 训练时优先用 Mini-batch，学习率和正则系数一起调

### 14.11 Hinge 损失与线性分类间隔

你可以把这部分理解为“0-1 损失的可优化替代”。

#### 1) 0-1 损失与 Hinge 损失的关系

0-1 损失只关心“分对还是分错”：

$$
\ell_{0-1}(y,\hat y)=\mathbf{1}(y\ne \hat y)
$$

它不连续、不可导，难以直接用梯度法优化。

在线性分类里，常把标签写成 $y\in\{-1,+1\}$，打分函数为：

$$
f(x)=w^T\phi(x)+b
$$

Hinge 损失定义为：

$$
\ell_{\text{hinge}}(x,y)=\max\big(0,\,1-yf(x)\big)
$$

它是 0-1 损失的上界（凸替代）。直观上：

- 若 $yf(x)\ge 1$，损失为 0（分对且“有余量”）
- 若 $0<yf(x)<1$，损失在 $(0,1)$（分对但离边界太近）
- 若 $yf(x)\le 0$，损失 $\ge 1$（分错，惩罚更大）

所以 hinge 的核心思想是：不仅要分对，还要“分得有把握”。

#### 2) 线性分类模型中的“间隔（margin）”

决策边界是：

$$
w^T\phi(x)+b=0
$$

常见两个间隔概念：

1. 函数间隔（functional margin）

$$
\gamma_i^{(f)}=y_i\big(w^T\phi(x_i)+b\big)=y_if(x_i)
$$

2. 几何间隔（geometric margin）

$$
\gamma_i^{(g)}=\frac{y_i\big(w^T\phi(x_i)+b\big)}{\|w\|}
$$

其中几何间隔可以理解为“样本点到超平面的带符号距离”。

#### 3) 为什么 hinge 里会出现常数 1

在线性 SVM 的标准形式里，约束写作：

$$
y_i\big(w^T\phi(x_i)+b\big)\ge 1
$$

当这个条件满足时，样本处在“安全区”；不满足就会被 hinge 惩罚。于是损失自然写成 $\max(0,1-yf(x))$。

#### 4) 为什么 hinge 和预测要共用同一个 $W$

你老师强调这点非常关键。核心是：

- 预测时我们用分数函数 $f(x)=w^T\phi(x)+b$ 决策
- hinge 损失也是惩罚这个同一个分数函数：$\max(0,1-yf(x))$

所以训练目标和预测规则必须对同一组参数起作用。

如果不用同一个 $W$，会出现逻辑断裂：

- 你最小化的是一套参数（记作 $W_{\text{loss}}$）对应的损失
- 但预测时用另一套参数（记作 $W_{\text{pred}}$）输出类别

这样训练就不能保证“损失下降会让预测变好”，优化目标与最终任务脱节。

从数学上看，梯度更新是：

$$
W \leftarrow W - \eta\,\frac{\partial \mathcal{L}}{\partial W}
$$

这个 $W$ 必须正是预测函数里的参数，否则更新没有实际意义。

一句话：

- 预测函数定义“模型做什么”
- 损失函数定义“模型错了怎么罚”
- 两者共用同一组参数，训练才有可解释性和有效性

#### 5) 用矩阵形式统一看分类（更适合考试）

当 $\phi(x)\in\mathbb{R}^d$：

1. 二分类（可写成一个向量）

- $w\in\mathbb{R}^d,\ b\in\mathbb{R}$
- 分数：$f(x)=w^T\phi(x)+b$
- 预测：$\hat y=\mathrm{sign}(f(x))$
- 损失（hinge）：$\ell=\max(0,1-yf(x))$

2. 多分类（更常写成矩阵）

- $W\in\mathbb{R}^{K\times d},\ b\in\mathbb{R}^K$
- 分数向量：$s(x)=W\phi(x)+b$
- 第 $k$ 类分数：$s_k(x)=w_k^T\phi(x)+b_k$
- 预测：$\hat y=\arg\max_k s_k(x)$

常见多分类损失：

- softmax 交叉熵（概率模型）
- multiclass hinge（间隔模型）

这两类损失虽然形式不同，但都会直接作用在同一个 $W$ 上。

#### 6) 结合你老师记法的小例子

如果老师写 $\phi(x)=[x_1,x_2]^T$，那就是“恒等映射”特例（不做额外变换）。

此时二分类分数就是：

$$
f(x)=w_1x_1+w_2x_2+b
$$

若要引入非线性特征，也可以定义：

$$
\phi(x)=[x_1,x_2,x_1^2,x_1x_2]^T
$$

公式仍然保持线性形式 $f(x)=w^T\phi(x)+b$，只是特征更丰富。

#### 7) 一句话总结

- 0-1 损失只区分“对/错”；hinge 进一步要求“间隔要足够大”。
- 线性分类中的间隔，本质是样本离决策边界的安全距离。
- 训练损失与预测函数共用同一组 $W$，是“学得好并且用得上”的必要条件。

### 14.12 感知机（Perceptron）笔记

感知机是最经典的线性二分类算法之一，可以看作很多现代分类方法的起点。

#### 1) 模型形式

仍使用你老师这套记法：

$$
f(x)=w^T\phi(x)+b,
\qquad
\hat y=\mathrm{sign}(f(x))
$$

其中标签通常取 $y\in\{-1,+1\}$。

#### 2) 感知机损失（只惩罚分错点）

感知机的单样本损失可写成：

$$
\ell_{\text{perc}}(x_i,y_i)=\max\big(0,\,-y_i f(x_i)\big)
$$

解释：

- 若 $y_if(x_i)>0$，说明分类正确，损失为 0
- 若 $y_if(x_i)\le 0$，说明分类错误（或落在边界上），损失为正

它和 hinge 的区别：

- perceptron 要求“分对就行”（阈值是 0）
- hinge 要求“分对且间隔至少 1”（阈值是 1）

#### 3) 经典更新规则

对一个被错分样本 $(x_i,y_i)$，参数更新：

$$
w \leftarrow w + \eta y_i\phi(x_i),
\qquad
b \leftarrow b + \eta y_i
$$

这里 $\eta>0$ 是学习率。

这个更新可以这样理解：

- 若样本是正类却被分错（$y_i=+1$），就把 $w$ 往 $+\phi(x_i)$ 方向推
- 若样本是负类却被分错（$y_i=-1$），就把 $w$ 往 $-\phi(x_i)$ 方向推

目的是让该样本下一次更可能被分对。

#### 4) 训练流程（可背）

1. 初始化 $w,b$（通常全 0）
2. 遍历训练样本
3. 若 $y_i(w^T\phi(x_i)+b)\le 0$，按上式更新
4. 重复多轮，直到没有错分或达到最大轮数

#### 5) 收敛性质（考试高频）

感知机收敛定理要点：

- 若数据线性可分，感知机在有限步内收敛
- 若数据线性不可分，感知机会持续波动，可能不收敛

因此工程上常见做法：

- 设最大 epoch
- 配合平均感知机（Averaged Perceptron）增强稳定性

#### 6) 与 SVM / Hinge 的关系

- 相同点：都基于线性打分 $f(x)=w^T\phi(x)+b$
- 不同点：
    - 感知机只关心分错与否，不显式最大化间隔
    - SVM + hinge 追求大间隔，泛化通常更稳健

可以记成：

- perceptron：先把分类做对
- SVM：不仅做对，还要离边界更远

#### 7) 小例子（二维）

设 $\phi(x)=[x_1,x_2]^T$，初始 $w=[0,0]^T,b=0,\eta=1$。

样本一：$x=[2,1]^T,\ y=+1$

- 当前 $f(x)=0$，视作错分
- 更新后 $w=[2,1]^T,\ b=1$

样本二：$x=[-1,-1]^T,\ y=-1$

- 此时 $f(x)=w^Tx+b=-2$，预测为负类，分对
- 不更新

这个例子体现了感知机的核心机制：只在错分时更新。

#### 8) 一句话总结

- 感知机是“错误驱动”的线性分类学习法。
- 它用同一个 $w,b$ 同时做预测和训练更新。
- 可分数据下能收敛，不可分数据下通常要靠改进版本和停止策略。