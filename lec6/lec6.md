# 第6讲：无监督学习、K-Means、PCA 与 GMM

> 根据 `transcript.md` 整理，并补充课堂中对应的算法推导与伪代码。

---

## 0. 本讲主线

本讲围绕无监督学习展开，核心可以概括成两条线：

- **化繁为简**：把高维、复杂的数据结构压缩成更简单、更易处理的形式
  - 聚类：把相似的数据放在一起
  - 降维：把冗余特征压缩掉
- **无中生有**：用生成模型去解释数据从哪里来
  - 例如高斯混合模型（GMM）

课堂一开始强调了一个最重要的事实：

- 监督学习依赖标签，目标是学出输入到输出的映射
- 无监督学习没有标签，目标是从特征中自动发现结构

这意味着无监督学习的关键不在“答案是什么”，而在于：

1. 用什么特征表示数据
2. 用什么相似度衡量数据之间的关系
3. 用什么模型把这些关系组织起来

---

## 1. 监督学习 vs 无监督学习

### 1.1 监督学习

监督学习的数据包含：

- 输入 $x$
- 标签 $y$

模型学习的是映射关系：

$$
f(x) \approx y
$$

例如图像分类里，数据不仅有形状、颜色等特征，还有人工标注的类别。

### 1.2 无监督学习

无监督学习只给输入，没有标签：

$$
\{x_1, x_2, \dots, x_n\}
$$

这时模型无法直接“照答案分类”，只能依赖特征本身去寻找规律。

课堂里提到的一个核心观点是：**相似性不是天然存在的，而是由特征和距离共同定义的。**

比如：

- 图像分割时，可以根据颜色、纹理、亮度等特征判断像素是否属于同一区域
- 分子分类时，可以根据 atomic features 判断原子或分子是否属于同类

所以，无监督学习的第一步往往不是算法本身，而是“特征怎么选、距离怎么算”。

---

## 2. 无监督学习的两类基本任务

### 2.1 聚类（Clustering）

聚类的目标是把相似样本分到同一组。

直观上，这就是课堂里说的“看形状、看颜色、看特征，把东西分开”。

### 2.2 降维（Dimensionality Reduction）

降维的目标是把高维数据映射到更低维空间，同时尽量保留数据的主要结构。

这类方法的用途包括：

- 数据压缩
- 可视化
- 去噪
- 作为后续聚类或分类的预处理

### 2.3 生成建模（Generative Modeling）

生成模型不只是“分组”，还要解释“数据是怎么生成出来的”。

高斯混合模型就是典型代表：它假设数据来自多个高斯分布的混合。

---

## 3. K-Means 聚类

K-Means 是最经典的聚类算法之一。它的核心思想非常直接：

- 先选几个中心点
- 每个样本找离自己最近的中心
- 再把中心更新成簇内样本的均值
- 重复直到稳定

### 3.1 优化目标

设数据集为

$$
X = \{x_1, x_2, \dots, x_n\}, \quad x_i \in \mathbb{R}^d
$$

要把数据分成 $K$ 类。令：

- $z_i \in \{1,2,\dots,K\}$ 表示样本 $x_i$ 的簇编号
- $\mu_k$ 表示第 $k$ 个簇的中心

K-Means 的目标函数是最小化簇内平方和：

$$
J = \sum_{i=1}^n \|x_i - \mu_{z_i}\|^2
$$

也可以写成：

$$
J = \sum_{k=1}^K \sum_{i:z_i=k} \|x_i - \mu_k\|^2
$$

这个目标的含义是：让每个样本都尽量靠近自己所属簇的中心。

### 3.2 交替优化

K-Means 不能一次性直接求出全局最优，所以采用交替优化：

1. 固定中心，更新簇分配
2. 固定簇分配，更新中心

#### 3.2.1 固定中心时，样本应该分给谁？

对每个样本 $x_i$，若中心固定，则应选择距离最近的中心：

$$
z_i = \arg\min_{k} \|x_i - \mu_k\|^2
$$

这一步就是“最近中心原则”。

#### 3.2.2 固定分配时，中心应该放在哪里？

对第 $k$ 个簇，记其中样本数为 $n_k$，目标为：

$$
J_k(\mu_k) = \sum_{i:z_i=k} \|x_i - \mu_k\|^2
$$

对 $\mu_k$ 求导：

$$
\frac{\partial J_k}{\partial \mu_k}
= \sum_{i:z_i=k} 2(\mu_k - x_i)
= 2n_k \mu_k - 2\sum_{i:z_i=k} x_i
$$

令导数为 0，得到：

$$
\mu_k = \frac{1}{n_k} \sum_{i:z_i=k} x_i
$$

也就是说：**中心就是该簇样本的均值。**

### 3.3 K-Means 伪代码

```python
def k_means(X, K, max_iter=100, tol=1e-6):
	# X: n x d data matrix
	# K: number of clusters
	# return: cluster labels and centroids

	mu = init_centroids(X, K)  # random init or k-means++

	for _ in range(max_iter):
		# assign each point to the nearest centroid
		z = []
		for x in X:
			k = argmin_k(distance_squared(x, mu[k]))
			z.append(k)

		# update centroids as cluster means
		new_mu = []
		for k in range(K):
			cluster_points = [x for x, label in zip(X, z) if label == k]
			if len(cluster_points) == 0:
				new_mu.append(reinit_one_centroid(X))
			else:
				new_mu.append(mean(cluster_points))

		# stop if centroids barely move
		if max_norm(new_mu - mu) < tol:
			break

		mu = new_mu

	return z, mu
```

### 3.4 K-Means 的性质

- **优点**：简单、速度快、容易实现
- **缺点**：
  - 对初始中心敏感
  - 只保证收敛到局部最优
  - 对异常点敏感
  - 假设簇形状大致是“球状”的

### 3.5 实际使用时的注意事项

- 特征尺度差异很大时，通常要先标准化，否则欧氏距离会被大尺度特征主导
- K 的选择通常结合肘部法、轮廓系数、业务经验
- 实践中常用 K-Means++ 改善初始化

---

## 4. PCA：主成分分析

PCA 是本讲最重要的数学部分。它的目标是：

- 找到一组新的正交坐标轴
- 让数据在这些方向上的投影尽可能“重要”
- 常见理解有两种：
  - 最大化投影方差
  - 最小化重构误差

这两种理解是等价的。

### 4.1 预处理：中心化

给定数据矩阵：

$$
X = [x_1^T; x_2^T; \dots; x_n^T] \in \mathbb{R}^{n \times d}
$$

先计算均值：

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i
$$

中心化后的数据为：

$$
\widetilde{x}_i = x_i - \bar{x}
$$

记中心化后的矩阵为 $X_c$。

为什么要中心化？因为 PCA 关心的是“围绕均值的波动结构”，而不是整体平移。

### 4.2 单个主成分的推导：最大化投影方差

设我们要找一个单位向量 $w$，把数据投影到这条直线上：

$$
y_i = w^T \tilde{x}_i, \quad \|w\|=1
$$

投影后的方差是：

$$
\operatorname{Var}(y)
= \frac{1}{n} \sum_{i=1}^n (w^T \tilde{x}_i)^2
$$

把它展开成矩阵形式：

$$
\operatorname{Var}(y)
= w^T \left(\frac{1}{n} \sum_{i=1}^n \tilde{x}_i \tilde{x}_i^T \right) w
$$

定义协方差矩阵：

$$
S = \frac{1}{n} X_c^T X_c
$$

于是目标变成：

$$
\max_{\|w\|=1} w^T S w
$$

这就是一个标准的 Rayleigh quotient 最大化问题。

#### 4.2.1 拉格朗日乘子法

构造拉格朗日函数：

$$
\mathcal{L}(w, \lambda) = w^T S w - \lambda (w^T w - 1)
$$

对 $w$ 求导并令其为 0：

$$
\frac{\partial \mathcal{L}}{\partial w}
= 2Sw - 2\lambda w = 0
$$

因此：

$$
Sw = \lambda w
$$

这说明最优方向一定是协方差矩阵的特征向量。

又因为目标是最大化 $w^T S w$，所以应选择**最大特征值对应的特征向量**作为第一主成分。

### 4.3 多个主成分

若要降到 $k$ 维，设主成分方向组成矩阵：

$$
W = [w_1, w_2, \dots, w_k] \in \mathbb{R}^{d \times k}
$$

并要求这些方向两两正交：

$$
W^T W = I
$$

投影后的总方差为：

$$
J(W) = \operatorname{tr}(W^T S W)
$$

目标变成：

$$
\max_{W^T W = I} \operatorname{tr}(W^T S W)
$$

其最优解仍然是：**取协方差矩阵最大的 $k$ 个特征值对应的特征向量。**

### 4.4 为什么这等价于最小化重构误差

把中心化样本投影到低维空间，再投回原空间：

$$
\hat{x}_i = W W^T \tilde{x}_i
$$

重构误差为：

$$
\sum_{i=1}^n \|\tilde{x}_i - \hat{x}_i\|^2
= \|X_c - X_c W W^T\|_F^2
$$

PCA 的另一个等价目标就是最小化这个误差。

直觉上，这表示：

- 只保留最重要的信息
- 丢掉的部分尽量少

由于 $W$ 的列向量正交，投影部分和残差部分是互补的，所以“最大化保留的方差”和“最小化损失的重构误差”本质上是同一件事。

### 4.5 特征值分解与 SVD

若协方差矩阵分解为：

$$
S = U \Lambda U^T
$$

其中 $\Lambda$ 的对角线是按降序排列的特征值：

$$
\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d
$$

那么：

- 前 $k$ 个主成分就是 $U$ 的前 $k$ 列
- 第 $j$ 个主成分对应的方差就是 $\lambda_j$

如果直接对中心化矩阵做 SVD：

$$
X_c = U \Sigma V^T
$$

则 PCA 方向就是 $V$ 的列向量。

这是工程实现里常用的做法，因为数值稳定性更好。

### 4.6 PCA 的解释方差比

保留前 $k$ 个主成分时，解释方差比为：

$$
\mathrm{EVR}(k) = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}
$$

它表示前 $k$ 个主成分一共解释了多少原始数据方差。

实际中常用它来决定降到几维。

### 4.7 PCA 伪代码

```python
def pca(X, k):
	# X: n x d data matrix
	# k: target dimension

	# 1. center data
	x_mean = mean(X, axis=0)
	Xc = X - x_mean

	# 2. compute covariance matrix (or use SVD directly)
	S = (Xc.T @ Xc) / len(X)

	# 3. eigen decomposition
	eigvals, eigvecs = eig(S)

	# 4. sort by descending eigenvalues
	idx = argsort(eigvals, descending=True)
	W = eigvecs[:, idx[:k]]

	# 5. project to low dimension
	Z = Xc @ W

	# 6. optional reconstruction
	X_hat = Z @ W.T + x_mean

	return Z, W, X_hat
```

### 4.8 PCA 的实践注意事项

- 如果不同特征量纲差别很大，通常先做标准化
- PCA 是线性的，只能捕捉线性主方向
- 若数据存在强非线性结构，PCA 可能不够，需要其他方法

---

## 5. 高斯混合模型（GMM）与 EM

课堂里提到“混合”时，对应的典型模型就是高斯混合模型。

它的思想不是把点硬塞进一个类，而是假设数据是由多个高斯分布共同生成的。

### 5.1 模型定义

设有 $K$ 个高斯分量，则数据分布写作：

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)
$$

其中：

- $\pi_k \ge 0$，且 $\sum_{k=1}^K \pi_k = 1$
- $\mu_k$ 是第 $k$ 个高斯的均值
- $\Sigma_k$ 是协方差矩阵

引入隐变量 $z_i$ 表示第 $i$ 个样本来自哪个分量：

$$
p(z_i = k) = \pi_k
$$

### 5.1.1 高斯分布基础

高斯分布（Gaussian distribution）是最常见的连续概率分布之一。单变量形式为：

$$
\mathcal{N}(x \mid \mu, \sigma^2)
= \frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

多变量形式为：

$$
\mathcal{N}(x \mid \mu, \Sigma)
= \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}}
\exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)
$$

其中：

- $\mu$ 控制中心位置
- $\sigma^2$ 或 $\Sigma$ 控制离散程度
- $\Sigma$ 还刻画不同维度之间的相关性

如果只用一个高斯分布来描述数据，本质上就是用一个“椭圆形”的概率模型去拟合一团样本。

对于单个高斯模型，最常见的参数估计就是样本均值和协方差：

$$
\hat{\mu} = \frac{1}{n} \sum_{i=1}^n x_i
$$

$$
\hat{\Sigma} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})(x_i - \hat{\mu})^T
$$

### 5.1.2 从单高斯到 GMM

GMM 可以看成单高斯模型的扩展：

- 单高斯：一团数据只用一个分布描述
- GMM：一团数据由多个高斯分量共同描述

因此，GMM 的本质是把“单峰分布”推广成“多峰混合分布”，每个分量负责解释一部分数据结构。

### 5.2 为什么要用 EM

EM 是 **Expectation-Maximization** 的缩写，中文一般叫**期望最大化算法**。

它适用于这类问题：

- 数据里有**隐变量**或**缺失信息**
- 直接最大化似然函数很困难
- 但如果隐变量已知，参数更新会变得很简单

在 GMM 里，隐变量就是每个样本属于哪个高斯分量。因为这个分量标签看不见，所以不能直接做完整监督学习，只能迭代估计。

EM 的核心思想是：

1. 先根据当前参数，估计隐变量的分布
2. 再利用这个估计结果更新参数

所以它本质上是一个“**先估计隐藏信息，再优化参数**”的交替迭代算法。

它通常优化的是对数似然：

$$
\log p(X \mid \theta)
$$

但由于含有隐变量 $Z$，直接最大化很麻烦，因此改成迭代求解。

一个重要性质是：**每次 EM 迭代都不会降低似然值**，所以它通常单调收敛到局部最优解。

### 5.3 E 步：责任度

定义责任度：

$$
\gamma_{ik} = p(z_i = k \mid x_i)
$$

根据贝叶斯公式：

$$
\gamma_{ik}
= \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
$$

这表示样本 $x_i$ 属于第 $k$ 个分量的“软概率”。

这一步叫 **E-step**，即 **Expectation step**。它做的事情不是直接改参数，而是计算“当前参数下，隐变量最可能是什么”。

从直觉上说：

- 如果一个样本更像某个高斯分量，那么它对这个分量的责任度就更高
- 所有分量的责任度加起来为 1

### 5.4 M 步：更新参数

先定义有效样本数：

$$
N_k = \sum_{i=1}^n \gamma_{ik}
$$

然后更新：

$$
\pi_k^{new} = \frac{N_k}{n}
$$

$$
\mu_k^{new} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} x_i
$$

$$
\Sigma_k^{new} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (x_i - \mu_k^{new})(x_i - \mu_k^{new})^T
$$

这一步叫 **M-step**，即 **Maximization step**。它的作用是：在 E-step 得到的软分配基础上，重新求出最优参数。

也就是说：

- E-step 负责“估计隐变量”
- M-step 负责“优化参数”

两步交替进行，直到收敛。

### 5.5 GMM 伪代码

```python
def gmm_em(X, K, max_iter=100, tol=1e-6):
	# initialize parameters
	pi, mu, Sigma = init_gmm_params(X, K)

	for _ in range(max_iter):
		# E-step: compute responsibilities
		gamma = zeros((len(X), K))
		for i, x in enumerate(X):
			denom = 0.0
			for k in range(K):
				gamma[i, k] = pi[k] * gaussian_pdf(x, mu[k], Sigma[k])
				denom += gamma[i, k]
			gamma[i, :] /= denom

		# M-step: update parameters
		Nk = gamma.sum(axis=0)
		new_pi = Nk / len(X)
		new_mu = [(gamma[:, k][:, None] * X).sum(axis=0) / Nk[k] for k in range(K)]
		new_Sigma = []
		for k in range(K):
			diff = X - new_mu[k]
			cov = (gamma[:, k][:, None, None] * \
				   (diff[:, :, None] @ diff[:, None, :])).sum(axis=0) / Nk[k]
			new_Sigma.append(cov)

		# convergence check
		if converged(pi, mu, Sigma, new_pi, new_mu, new_Sigma, tol):
			break

		pi, mu, Sigma = new_pi, new_mu, new_Sigma

	return pi, mu, Sigma, gamma
```

### 5.6 GMM 的性质

- **软聚类**：每个样本对各个簇都有概率
- **椭球形簇**：协方差矩阵允许各方向不同尺度
- **表达能力比 K-Means 更强**：能拟合更复杂的簇形状
- **缺点**：
  - 对初始化敏感
  - 可能陷入局部最优
  - 计算量通常比 K-Means 更大

### 5.7 EM 的直观理解

可以把 EM 理解成一个“先分锅、再炒菜”的过程：

- 先根据当前模型判断每个样本更像哪一锅，也就是 E-step
- 再根据这个判断重新估计每一锅的配方，也就是 M-step

它不是一次性求出全局最优，而是通过反复迭代，让模型越来越好。

### 5.8 K-Means 与 GMM 的关系

两者可以看成是同一思想的不同版本：

- K-Means：硬分配，中心是均值，簇形状近似球形
- GMM：软分配，允许不同协方差，簇形状可以是椭球形

可以把 K-Means 理解成 GMM 在某些极限条件下的特例。

---

## 6. 三个算法的统一理解

| 算法 | 任务 | 输出 | 关键思想 | 优点 | 局限 |
|---|---|---|---|---|---|
| K-Means | 聚类 | 簇标签、中心 | 最近中心 + 均值更新 | 简单快速 | 只能处理近似球形簇 |
| PCA | 降维 | 低维表示、投影矩阵 | 最大方差 / 最小重构误差 | 数学清晰、可解释性强 | 只能学线性结构 |
| GMM | 生成建模 / 软聚类 | 混合参数、责任度 | 概率生成 + EM | 表达能力强 | 计算更重、易陷局部最优 |

### 6.1 本讲最重要的结论

1. **无监督学习的核心不是“有无标签”，而是“如何定义结构”**
2. **K-Means 解决的是“怎么分组”**
3. **PCA 解决的是“怎么压缩信息”**
4. **GMM 解决的是“数据怎么生成、怎么软分配”**

---

## 7. 课堂开场内容的整理版

把转录里的零散表述整理成更正式的版本，大意如下：

- 数据的属性或个体特征可以统称为 features
- 无监督学习会基于这些特征去做划分
- 划分的依据可以是颜色、形状、距离，或者更抽象的统计结构
- 聚类和降维是最典型的两类无监督任务
- 图像分割可以用 K-Means，主成分分析用于压缩和提取主方向，混合模型用于描述更复杂的数据生成机制

---

## 8. 小结

本讲可以用一句话概括：

**在没有标签的情况下，先用特征定义相似性，再用聚类、降维和生成模型去发现数据结构。**

其中：

- K-Means 负责把相似样本放在一起
- PCA 负责找出最重要的投影方向
- GMM 负责用概率模型解释“数据从哪里来”

如果只记住一条数学结论，PCA 的核心就是：

$$
\max_{\|w\|=1} w^T S w
\quad \Longrightarrow \quad
Sw = \lambda w
$$

也就是：**主成分就是协方差矩阵的特征向量，按特征值从大到小选择。**
