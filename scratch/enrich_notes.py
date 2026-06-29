# -*- coding: utf-8 -*-
import os
import re

# Concrete content for each lecture to enrich the HTML notes
lecture_enrichments = {
    "lec1": {
        "code": """# -*- coding: utf-8 -*-
# 第 1 讲: 绪论 - 人工智能系统工程闭环的极简 NumPy 实现
import numpy as np

# 1. 任务定义 (Task): 线性分类任务 (2D 点分类，判别是类别 0 还是 1)
# 输入: x = [x1, x2] ∈ R^2
# 输出: y ∈ {0, 1}
X = np.array([[1.0, 2.0], [2.0, 1.0], [2.0, 3.0], [3.0, 1.0]]) # 数据 D
Y = np.array([0, 0, 1, 1])

# 2. 模型与参数 (Model & Representation): 线性分类器 + Sigmoid 激活
# f_theta(x) = sigmoid(w^T * x + b)
w = np.array([0.5, -0.5])
b = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(x):
    return sigmoid(np.dot(x, w) + b)

# 3. 优化与学习 (Optimization): 单步梯度下降更新
# Loss = - [y * log(p) + (1-y) * log(1-p)] (交叉熵损失)
# 更新规则: w <- w - alpha * (p - y) * x
alpha = 0.1 # 学习率
print("--- 初始状态 ---")
for x, y in zip(X, Y):
    p = forward(x)
    loss = - (y * np.log(p) + (1-y) * np.log(1-p))
    print(f"输入: {x}, 目标: {y}, 预测概率: {p:.4f}, 损失: {loss:.4f}")

# 执行单步梯度更新
grad_w = np.zeros_like(w)
grad_b = 0.0
for x, y in zip(X, Y):
    p = forward(x)
    grad_w += (p - y) * x
    grad_b += (p - y)
w -= alpha * grad_w / len(X)
b -= alpha * grad_b / len(X)

print("\\n--- 梯度下降一步后参数 ---")
print(f"更新后的 w: {w}, b: {b}")

# 4. 评价指标 (Evaluation): 准确率 (Accuracy)
predictions = np.array([1 if forward(x) >= 0.5 else 0 for x in X])
accuracy = np.mean(predictions == Y)
print(f"训练集准确率: {accuracy * 100:.1f}%")
""",
        "flow_table": """<tr><td>现实任务</td><td>视觉目标感知与交互规划。输入为图像/视频数据，输出为像素分类/动作决策。失败于光照突变、未见分布。</td></tr>
<tr><td>输入/输出定义</td><td>输入 x ∈ R^{H×W×3} 图像，输出 y ∈ R^{H×W} 像素掩码。参数为网络权重 θ。失败于标注噪声、语义模糊。</td></tr>
<tr><td>数据与表示</td><td>将物理图像映射成像素特征及多维向量表示 φ(x)。失败于特征表达不充分、维度灾难。</td></tr>
<tr><td>模型或规则</td><td>选择感知映射函数 fθ(x) = P(y|x)。利用数据分布驱动端到端参数优化。失败于结构归纳偏置不匹配。</td></tr>
<tr><td>训练/推理</td><td>基于损失函数 L(y, fθ(x)) 使用 Adam 优化。推理时前向传播计算预测值。失败于局部最优或梯度消失。</td></tr>
<tr><td>评价与迭代</td><td>在测试集上计算 mAP、mIoU 和 Accuracy。根据误差分析（如漏检、错检）调整模型结构和数据增强。</td></tr>"""
    },
    "lec2": {
        "code": """# -*- coding: utf-8 -*-
# 第 2 讲: 逻辑与知识表示 - 简单命题推理与知识图谱路径检索的纯 Python 实现
class SimpleKnowledgeGraph:
    def __init__(self):
        # 知识表示：用邻接表存储关系三元组 (头实体, 关系, 尾实体)
        self.triples = []
        self.edges = {}

    def add_triple(self, h, r, t):
        self.triples.append((h, r, t))
        if h not in self.edges:
            self.edges[h] = []
        self.edges[h].append((r, t))

    # 推理任务：基于路径的推理 (Path-based Reasoning / Relational Path Search)
    # 输入: 起点 head, 关系路径 path (关系的列表)
    # 输出: 终点集合
    def path_query(self, head, path):
        current_entities = {head}
        for step_relation in path:
            next_entities = set()
            for entity in current_entities:
                if entity in self.edges:
                    for relation, target in self.edges[entity]:
                        if relation == step_relation:
                            next_entities.add(target)
            current_entities = next_entities
            if not current_entities:
                break
        return list(current_entities)

# 1. 知识表示构建
kg = SimpleKnowledgeGraph()
kg.add_triple("人工智能", "属于学科", "计算机科学")
kg.add_triple("计算机科学", "包含子域", "机器学习")
kg.add_triple("机器学习", "基础技术", "深度学习")
kg.add_triple("深度学习", "典型架构", "Transformer")

# 2. 路径推理演示
# 查询：\"人工智能\" 沿着 ['属于学科', '包含子域', '基础技术', '典型架构'] 的推理结果
relation_path = ["属于学科", "包含子域", "基础技术", "典型架构"]
inferred_targets = kg.path_query("人工智能", relation_path)

print("--- 知识图谱路径推理 ---")
print(f"推理路径: 人工智能 -> {' -> '.join(relation_path)}")
print(f"推理得到的尾实体: {inferred_targets}") # 应输出 ['Transformer']
""",
        "flow_table": """<tr><td>现实知识库</td><td>建立现实世界实体及关联的形式化表示。输入为非结构化文本，输出为结构化三元组。失败于自然语言多义性。</td></tr>
<tr><td>逻辑符号化</td><td>将自然命题转换为命题公式或一阶谓词逻辑表达式。输入为自然命题，输出为符号公式。失败于表达力不足。</td></tr>
<tr><td>规则库构建</td><td>定义产生式规则或推理机规则（如 IF A AND B THEN C）。输出为规则集合。失败于规则冲突与维护成本高。</td></tr>
<tr><td>图谱实体抽取</td><td>通过 NER 提取实体，通过关系抽取构建三元组关系 (h, r, t)。失败于实体指代消解失败、关系混淆。</td></tr>
<tr><td>路径推理计算</td><td>使用 PRA (Path Ranking Algorithm) 或逻辑消解进行多跳推理计算。失败于图结构稀疏、组合爆炸。</td></tr>
<tr><td>知识库评估</td><td>计算图谱完备度、三元组准确率以及逻辑一致性检测。失败于假阳性三元组污染知识库。</td></tr>"""
    },
    "lec3": {
        "code": """# -*- coding: utf-8 -*-
# 第 3 讲: 搜索求解 - A* 搜索算法解决 2D 网格寻路问题的极简 Python 实现
import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
    def empty(self):
        return len(self.elements) == 0
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    def get(self):
        return heapq.heappop(self.elements)[1]

# 启发式函数 (Heuristic): 曼哈顿距离
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

# A* 算法实现
# grid: 0 为通路，1 为障碍物
def a_star_search(grid, start, goal):
    height, width = len(grid), len(grid[0])
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break

        # 4 方向探索
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_node = (current[0] + dx, current[1] + dy)
            if 0 <= next_node[0] < height and 0 <= next_node[1] < width:
                if grid[next_node[0]][next_node[1]] == 1: # 障碍物
                    continue
                new_cost = cost_so_far[current] + 1 # 边权为 1
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(goal, next_node)
                    frontier.put(next_node, priority)
                    came_from[next_node] = current

    # 重建路径
    current = goal
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path, cost_so_far.get(goal, float('inf'))

# 1. 地图构建 (5x5，中间有一道墙)
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
]
start, goal = (0, 0), (4, 4)

# 2. 执行搜索
path, cost = a_star_search(grid, start, goal)
print("--- A* 搜索求解结果 ---")
print(f"起点: {start}, 终点: {goal}")
print(f"求解路径: {path}")
print(f"路径总代价 (步数): {cost}")
""",
        "flow_table": """<tr><td>现实问题定义</td><td>定义问题的状态空间 S、初始状态 s₀、动作集合 A、目标测试 Goal(s) 和路径代价 c(s, a, s')。失败于动作过多导致分支因子爆炸。</td></tr>
<tr><td>图搜索树构建</td><td>维护前沿节点表 Frontier 和已探索表 Explored。防止环路和重复搜索。失败于内存溢出（如未剪枝的广度搜索）。</td></tr>
<tr><td>启发式评估设计</td><td>设计启发函数 h(s)。必须满足可采纳性（Admissibility, h(s) <= h*(s)）和一致性（Consistency）。失败于 h(s) 过高估计导致解非最优。</td></tr>
<tr><td> Frontier 节点排序</td><td>利用评估函数 f(s) = g(s) + h(s) 对 Frontier 队列进行堆排序，保证最优节点优先出队。失败于堆维护开销随规模指数增加。</td></tr>
<tr><td>状态转移与解抽取</td><td>从队列中取出目标节点，反向回溯 came_from 链生成最短路径解。失败于状态空间不连通导致无解。</td></tr>
<tr><td>完备与最优性评估</td><td>分析算法在具体分支因子 b、深度 d 上的时间与空间复杂度。验证在有限图上的完备性。失败于搜索死循环。</td></tr>"""
    },
    "lec4": {
        "code": """# -*- coding: utf-8 -*-
# 第 4 讲: 线性模型 - NumPy 实现 Logistic 回归梯度下降与交叉熵损失的完整实现
import numpy as np

# 1. 构造数据集 (线性可分数据)
np.random.seed(42)
num_samples = 100
X = np.random.randn(num_samples, 2)
# 真实的超平面 w_true = [1.5, -2.0], b_true = 0.5
y_true = (X[:, 0] * 1.5 - X[:, 1] * 2.0 + 0.5 > 0).astype(int)

# 2. 模型定义
w = np.zeros(2)
b = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -15, 15)))

# 3. 训练循环 (梯度下降)
lr = 0.1
epochs = 200

for epoch in range(epochs):
    # 前向预测
    z = np.dot(X, w) + b
    p = sigmoid(z)
    
    # 计算交叉熵损失 Loss = -1/N * sum( y*log(p) + (1-y)*log(1-p) )
    loss = -np.mean(y_true * np.log(p + 1e-15) + (1 - y_true) * np.log(1 - p + 1e-15))
    
    # 梯度计算: dL/dw = 1/N * X^T * (p - y), dL/db = 1/N * sum(p - y)
    dw = np.dot(X.T, (p - y_true)) / num_samples
    db = np.sum(p - y_true) / num_samples
    
    # 参数更新
    w -= lr * dw
    b -= lr * db
    
    if epoch % 40 == 0:
        accuracy = np.mean((p >= 0.5).astype(int) == y_true)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Accuracy: {accuracy*100:.1f}%")

print("\\n--- 最终优化参数 ---")
print(f"估计 w: {w}, 估计 b: {b}")
print(f"真实 w_true: [1.5, -2.0], b_true: 0.5")
""",
        "flow_table": """<tr><td>现实数据输入</td><td>输入高维特征矩阵 X ∈ R^{N×d} 和一维标签向量 Y ∈ R^N。进行均值归一化，防止量纲差异。失败于特征多重共线性。</td></tr>
<tr><td>线性映射构建</td><td>构建前向预测公式 z = Xw + b。若为回归，预测值为 z；若为分类，将 z 输入 Sigmoid 转换为概率。失败于非线性分布无法拟合。</td></tr>
<tr><td>损失目标度量</td><td>定义优化目标。线性回归使用 MSE (均方误差)；Logistic 回归使用 Cross Entropy (交叉熵损失)。失败于异常值大幅拉偏 MSE。</td></tr>
<tr><td>梯度计算推导</td><td>对参数 w 和 b 求偏导。使用解析公式计算批量梯度或随机梯度。失败于梯度消失（Sigmoid 饱和区）或梯度爆炸。</td></tr>
<tr><td>参数迭代更新</td><td>利用梯度下降更新参数：w <- w - η * ∇w。动态衰减学习率 η。失败于步长过大产生震荡不收敛。</td></tr>
<tr><td>收敛与泛化评估</td><td>计算训练集和测试集上的损失与精度，绘制学习曲线，检测是否过拟合或欠拟合。失败于过拟合测试集性能暴跌。</td></tr>"""
    },
    "lec5": {
        "code": """# -*- coding: utf-8 -*-
# 第 5 讲: SVM 与模型选择评估 - 铰链损失 (Hinge Loss) 与分类评估指标的 Python 计算
import numpy as np

# 1. 铰链损失 (Hinge Loss) 计算
# 公式: L_hinge(x) = max(0, 1 - y * f(x))
# y 是真实标签 (取值为 -1 或 +1)，f(x) 是模型预测的原始得分 (margin score)
def hinge_loss(y_true, f_x):
    return np.maximum(0, 1 - y_true * f_x)

# 2. 混淆矩阵与评估指标计算 (Accuracy, Precision, Recall, F1)
def calculate_metrics(y_true, y_pred):
    # 输入为二进制 0/1 标签
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}

# 真实值与模型预测 margin 分数
y_true_svm = np.array([1, -1, 1, -1, 1])
f_x_svm = np.array([1.2, 0.5, -0.3, -1.5, 0.1]) # 模型输出

losses = hinge_loss(y_true_svm, f_x_svm)
print("--- Hinge Loss 计算 ---")
for y, score, loss in zip(y_true_svm, f_x_svm, losses):
    print(f"真实标签: {y:2d} | 预测得分: {score:4.1f} | Hinge Loss: {loss:.4f}")

# 评估指标测试
y_true_cls = np.array([1, 0, 1, 1, 0, 0, 1, 0])
y_pred_cls = np.array([1, 0, 0, 1, 1, 0, 1, 0])
metrics = calculate_metrics(y_true_cls, y_pred_cls)

print("\\n--- 分类指标评估 ---")
print(f"混淆矩阵: TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
print(f"准确率 Accuracy : {metrics['Accuracy']*100:.1f}%")
print(f"精确率 Precision: {metrics['Precision']*100:.1f}%")
print(f"召回率 Recall   : {metrics['Recall']*100:.1f}%")
print(f"F1 评分 F1-Score: {metrics['F1-Score']:.4f}")
""",
        "flow_table": """<tr><td>样本空间投影</td><td>输入数据，选用核函数（线性、RBF等）将原始特征投影到高维希尔伯特空间。失败于核参数 σ 选择不当导致过拟合或欠拟合。</td></tr>
<tr><td>最大间隔边界构建</td><td>构建软间隔优化问题：min 1/2 ||w||² + C Σ ξᵢ。C 作为正则化超参数控制间隔大小与误分类容忍度。失败于 C 过大导致对噪点敏感。</td></tr>
<tr><td>对偶问题求解</td><td>利用拉格朗日乘子法转化为对偶问题。使用 SMO 算法交替优化乘子 α。失败于样本规模过大导致 O(N²) 的求解效率暴跌。</td></tr>
<tr><td>支持向量筛选</td><td>根据 KKT 条件筛选出支持向量（即对应拉格朗日乘子 α_i > 0 的样本点）。失败于不满足 Slater 约束。</td></tr>
<tr><td>决策面预测</td><td>利用 f(x) = sign(Σ α_i y_i K(x_i, x) + b) 进行测试分类。失败于支持向量过少导致泛化性能差。</td></tr>
<tr><td>模型评估与选择</td><td>使用 k 折交叉验证评估混淆矩阵、F1-Score、ROC/AUC 指标，根据偏差-方差折中调整参数。失败于指标盲目追求 Accuracy 而忽视了类别不平衡。</td></tr>"""
    },
    "lec6": {
        "code": """# -*- coding: utf-8 -*-
# 第 6 讲: 无监督学习 - NumPy 纯手工实现 K-Means 聚类与 PCA 降维
import numpy as np

# ================= K-Means 聚类 =================
def k_means(X, K, max_iter=20):
    # 1. 随机初始化 K 个中心点
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    for _ in range(max_iter):
        # 2. 分配步骤：计算样本到每个中心点的欧氏距离，并归属到最近中心
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        # 3. 更新步骤：中心点移动到对应簇样本的均值处
        new_centroids = np.array([X[labels == k].mean(axis=0) if len(X[labels == k]) > 0 else centroids[k] for k in range(K)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# ================= PCA 主成分分析 =================
def pca(X, k_components):
    # 1. 中心化数据
    X_mean = np.mean(X, axis=0)
    Xc = X - X_mean
    # 2. 计算协方差矩阵 S = 1/N * Xc.T @ Xc
    S = np.dot(Xc.T, Xc) / len(X)
    # 3. 特征值分解
    eigvals, eigvecs = np.linalg.eigh(S)
    # 4. 按特征值降序排序，取前 k 个特征向量
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:k_components]]
    # 5. 投影到低维空间
    Z = np.dot(Xc, W)
    return Z, W

# 测试数据: 6个二维数据点
X_toy = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0], [1.2, 1.5], [5.5, 8.5]])

# 执行 K-Means
labels, centroids = k_means(X_toy, K=2)
print("--- K-Means 结果 ---")
print(f"数据类别分配 labels: {labels}")
print(f"最终聚类中心 centroids:\\n{centroids}")

# 执行 PCA (降为 1 维)
Z, W = pca(X_toy, k_components=1)
print("\\n--- PCA 结果 ---")
print(f"投影后低维坐标 Z:\\n{Z}")
print(f"主成分投影矩阵 W:\\n{W}")
""",
        "flow_table": """<tr><td>无标签数据 X</td><td>输入样本集 X ∈ R^{N×d}，移除任何人工标签，分析其内在特征结构。失败于高维空间稀疏导致距离度量失效。</td></tr>
<tr><td>选择结构假设</td><td>K-Means 假设簇呈球状分布；PCA 假设信息主要保存在最大方差的线性子空间中；GMM 假设数据由 K 个高斯源混合生成。失败于非线性流形假设不匹配。</td></tr>
<tr><td>聚类/降维/生成建模</td><td>选择算法流程。降维取协方差矩阵特征向量；聚类交替分配与更新中心。失败于特征量纲不统一导致被大尺度特征霸占主导权。</td></tr>
<tr><td>优化隐变量</td><td>通过 EM 算法更新 GMM 责任度；通过 K-Means 更新硬分类标签；通过特征值分解计算主坐标。失败于局部最优解。</td></tr>
<tr><td>解释结构</td><td>评估聚类紧凑度（轮廓系数）或降维的累积解释方差比（EVR）。分析低维主成分含义。失败于低维度投影丢失判别性信息。</td></tr>
<tr><td>用于可视化或下游任务</td><td>将降维后的 Z ∈ R^{N×k} 输入到分类器中，或利用 TSNE/PCA 在 2D 空间绘制散点图。失败于过度降维导致下游分类精度雪崩。</td></tr>"""
    },
    "lec7": {
        "code": """# -*- coding: utf-8 -*-
# 第 7 讲: 神经网络基础 - 纯 NumPy 实现双层 MLP 的前向传播与反向传播梯度推导
import numpy as np

# 1. 初始化两层感知机 (MLP) 参数
np.random.seed(42)
d_in, d_hidden, d_out = 2, 3, 1
W1 = np.random.randn(d_in, d_hidden) * 0.1
b1 = np.zeros((1, d_hidden))
W2 = np.random.randn(d_hidden, d_out) * 0.1
b2 = np.zeros((1, d_out))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    # a 为 sigmoid 激活后的值
    return a * (1 - a)

# 单个样本前向传播与反向传播
x = np.array([[1.0, 2.0]]) # 1x2 输入
y_true = np.array([[1.0]]) # 1x1 目标值

# --- 1. 前向传播 ---
z1 = np.dot(x, W1) + b1     # 1x3
a1 = sigmoid(z1)            # 1x3 (隐藏层输出)
z2 = np.dot(a1, W2) + b2    # 1x1
y_pred = sigmoid(z2)        # 1x1 (输出层预测)

# 均方误差损失 Loss = 0.5 * (y_pred - y_true)^2
loss = 0.5 * np.sum((y_pred - y_true)**2)

# --- 2. 反向传播 (梯度推导) ---
# dLoss/dy_pred = y_pred - y_true
# dLoss/dz2 = (y_pred - y_true) * y_pred * (1 - y_pred)
dz2 = (y_pred - y_true) * sigmoid_derivative(y_pred) # 1x1

# dLoss/dW2 = a1.T @ dz2, dLoss/db2 = dz2
dW2 = np.dot(a1.T, dz2) # 3x1
db2 = dz2               # 1x1

# 误差回传到隐藏层：dLoss/da1 = dz2 @ W2.T
da1 = np.dot(dz2, W2.T) # 1x3
# dLoss/dz1 = da1 * a1 * (1 - a1)
dz1 = da1 * sigmoid_derivative(a1) # 1x3

# dLoss/dW1 = x.T @ dz1, dLoss/db1 = dz1
dW1 = np.dot(x.T, dz1) # 2x3
db1 = dz1              # 1x3

print("--- 双层 MLP 梯度推导测试 ---")
print(f"前向预测值 y_pred: {y_pred[0,0]:.4f} | 损失 loss: {loss:.4f}")
print(f"W2 的梯度 dW2:\\n{dW2}")
print(f"W1 的梯度 dW1:\\n{dW1}")
""",
        "flow_table": """<tr><td>输入特征 x</td><td>输入高维特征向量 x ∈ R^d。进行批归一化 (BatchNorm)，加速收敛。失败于特征未中心化。</td></tr>
<tr><td>层间线性变换</td><td>计算 z = Wx + b。实现线性降维或映射，参数量由 W 尺寸决定。失败于参数初始化不当产生饱和或退化。</td></tr>
<tr><td>非线性激活函数</td><td>输入 z 到激活函数 a = σ(z)（如 ReLU, Sigmoid）。引入非线性，打破叠加效应。失败于 ReLU 产生神经元坏死（Dying ReLU）。</td></tr>
<tr><td>前向多层传播</td><td>多层堆叠得到最终预测值 y_pred。构建高度复杂的层次化特征表征。失败于网络过深导致梯度消失。</td></tr>
<tr><td>误差反向传播</td><td>使用链式法则计算 dL/dWi 与 dL/dbi。将输出层误差沿计算图反向逐层回传。失败于梯度流中断或梯度爆炸。</td></tr>
<tr><td>正则化与参数更新</td><td>应用 L2 正则化（权重衰减）和 Dropout 防止过拟合。利用 Adam/SGD 迭代更新参数。失败于过拟合。</td></tr>"""
    },
    "lec8": {
        "code": """# -*- coding: utf-8 -*-
# 第 8 讲: 卷积神经网络 - 纯 NumPy 编写的 2D 互相关 (卷积前向) 运算与输出尺寸计算
import numpy as np

# 1. 2D 互相关运算 (2D Cross-correlation / Convolution Forward step)
# Input: X (H x W), Kernel: K (Kh x Kw), Padding: P, Stride: S
def conv2d_forward(X, K, padding=0, stride=1):
    H, W = X.shape
    Kh, Kw = K.shape
    
    # 应用 Padding (四周填充 0)
    X_padded = np.pad(X, padding, mode='constant', constant_values=0)
    Hp, Wp = X_padded.shape
    
    # 2. 计算输出维度 H_out = floor((Hp - Kh) / S) + 1
    H_out = (Hp - Kh) // stride + 1
    W_out = (Wp - Kw) // stride + 1
    
    Y = np.zeros((H_out, W_out))
    
    # 3. 互相关滑动计算
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + Kh
            w_start = j * stride
            w_end = w_start + Kw
            
            # 局部区域点乘求和
            region = X_padded[h_start:h_end, w_start:w_end]
            Y[i, j] = np.sum(region * K)
            
    return Y, (H_out, W_out)

# 测试数据
X_img = np.array([
    [1.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0]
])
K_filter = np.array([
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0]
])

Y_conv, shape_out = conv2d_forward(X_img, K_filter, padding=1, stride=1)
print("--- 2D 卷积滑动计算结果 ---")
print(f"输入形状: {X_img.shape} | 卷积核形状: {K_filter.shape}")
print(f"Padding: 1, Stride: 1 -> 输出形状: {shape_out}")
print(f"输出特征图 Y_conv:\\n{Y_conv}")
""",
        "flow_table": """<tr><td>图像输入</td><td>输入图像张量 X ∈ R^{N×C×H×W}。保存二维空间拓扑结构。失败于图像畸变未做对齐和归一化。</td></tr>
<tr><td>卷积核滑动计算</td><td>卷积核 K 在图像上以 Stride 滑动做点乘累加运算。实现局部连接和感受野扩展。失败于卷积核初始化不当导致输出全 0。</td></tr>
<tr><td>Padding/Stride 边界控制</td><td>利用填充 P 保持边界信息，通过步长 S 进行下采样压缩。输出维度计算：⌊(H+2P-K)/S⌋+1。失败于尺寸下采样过快导致信息大量丢失。</td></tr>
<tr><td>激活层与池化层</td><td>使用 ReLU 激活后，进入 MaxPool 或 AvgPool 降低分辨率。提升平移不变性与鲁棒性。失败于池化丢弃了细粒度空间位置信息。</td></tr>
<tr><td>堆叠骨干网络</td><td>通过堆叠 Conv-Pool 块并融合经典架构（如 ResNet 残差快）。由浅入深提取几何、语义特征。失败于深层退化。</td></tr>
<tr><td>全连接层输出</td><td>通过 GAP (全局平均池化) 或 Flatten，最后过 Linear 输出分类概率。参数量集中在此处。失败于特征过密导致过拟合。</td></tr>"""
    },
    "lec9": {
        "code": """# -*- coding: utf-8 -*-
# 第 9 讲: PyTorch 实践 - 完整可运行的神经网络训练循环 (包含 Dataset、Model、Optimizer 与 Loss)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. 构造 PyTorch 数据流 (Dataset & DataLoader)
torch.manual_seed(42)
X_data = torch.randn(120, 2)
Y_data = (X_data[:, 0] * 1.5 - X_data[:, 1] * 2.0 + 0.5 > 0).long()

dataset = TensorDataset(X_data, Y_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. 定义神经网络架构 (nn.Module)
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2) # 输出 2 类 logits
        )
    def forward(self, x):
        return self.net(x)

model = SimpleClassifier()

# 3. 损失函数与优化器 (Loss & Optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# 4. 经典训练循环
print("--- PyTorch 训练循环测试 ---")
epochs = 5
for epoch in range(epochs):
    model.train() # 训练模式
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()        # 1. 清空梯度
        outputs = model(x_batch)      # 2. 前向传播
        loss = criterion(outputs, y_batch) # 3. 计算损失
        loss.backward()              # 4. 反向传播
        optimizer.step()             # 5. 迭代更新
        
        # 统计
        epoch_loss += loss.item() * x_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        
    avg_loss = epoch_loss / total
    accuracy = correct / total
    print(f"Epoch {epoch+1:d}/{epochs:d} | Batch Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.1f}%")
""",
        "flow_table": """<tr><td> Tensor 定义与计算</td><td>创建张量，绑定数据类型 (dtype) 并通过 device 转移到 GPU，使用 autograd 准备追踪梯度。失败于内存/显存溢出。</td></tr>
<tr><td>Dataset与DataPipe</td><td>继承 Dataset 类重写 __getitem__ 和 __len__，使用 DataLoader 准备好多进程批取、打乱和加载。失败于数据管道阻塞延迟。</td></tr>
<tr><td>nn.Module网络声明</td><td>继承 nn.Module 类构建多层前向网络结构，声明可训练 Parameters，配置 dropout 与激活。失败于输入特征维度前后错配。</td></tr>
<tr><td>零参数前向推理</td><td>模型在 model.eval() 和 with torch.no_grad() 状态下执行前向推理，计算测试集的损失和指标。失败于忘开评估状态导致 BatchNorm 不准。</td></tr>
<tr><td>反向传播与优化</td><td>在前向计算输出后计算 loss，使用 optimizer.zero_grad() 清空以前累积梯度，调用 loss.backward() 产生新梯度。失败于梯度被错误累加。</td></tr>
<tr><td>Adam/SGD 参数更新</td><td>调用 optimizer.step() 更新参数。根据验证集表现应用学习率调度器。失败于模型未真正收敛时强行中断。</td></tr>"""
    },
    "lec10": {
        "code": """# -*- coding: utf-8 -*-
# 第 10 讲: 序列建模与注意力 - PyTorch 实现 RNN Cell 与单头 Scaled Dot-Product Attention 的张量维度演变
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 1. RNN Cell 单步推进 =================
batch_size, seq_len, input_size, hidden_size = 2, 4, 8, 16
rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# 输入序列 X (2 x 4 x 8)，隐藏状态初始化为 0 (2 x 16)
X = torch.randn(batch_size, seq_len, input_size)
h_t = torch.zeros(batch_size, hidden_size)

print("--- RNN Cell 隐藏状态演化 ---")
for t in range(seq_len):
    # 逐时刻推递，h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)
    h_t = rnn_cell(X[:, t, :], h_t)
    print(f"时刻 t={t} | 隐藏状态张量形状: {list(h_t.shape)}")

# ================= 2. 缩放点积注意力 (Scaled Dot-Product Attention) =================
# 假定输入序列通过线性变换得到了 Q, K, V (2 x 4 x 16)
# Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
Q = torch.randn(batch_size, seq_len, hidden_size)
K = torch.randn(batch_size, seq_len, hidden_size)
V = torch.randn(batch_size, seq_len, hidden_size)

d_k = K.size(-1)
# 1. 相似度计算: Batch Matmul (B x N x H) @ (B x H x N) -> B x N x N
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
# 2. 归一化为概率分布
attn_weights = F.softmax(scores, dim=-1)
# 3. 汇聚信息: (B x N x N) @ (B x N x H) -> B x N x H
output = torch.matmul(attn_weights, V)

print("\\n--- Attention 机制张量变化 ---")
print(f"注意力得分矩阵 scores 形状      : {list(scores.shape)}")
print(f"归一化注意力权重 weights 形状  : {list(attn_weights.shape)}")
print(f"注意力输出 output 形状         : {list(output.shape)}")
""",
        "flow_table": """<tr><td>序列数据准备</td><td>输入变长或等长文本序列，经由 Word Embedding 层映射成三维张量 X ∈ R^{B×T×d_model}。失败于未对填充 token 做 mask 处理。</td></tr>
<tr><td>RNN 循环步进</td><td>在循环中一步一步输入 x_t，计算 h_t = tanh(W_x x_t + W_h h_{t-1} + b)。失败于长距离依赖导致的梯度消失。</td></tr>
<tr><td>LSTM 门控传递</td><td>利用遗忘门 f_t、输入门 i_t、输出门 o_t，在 cell state 记忆管道上累加流转。缓解长序列衰减。失败于多门控乘法增加计算延迟。</td></tr>
<tr><td>Q/K/V 投影投射</td><td>将输入序列线性映射为 Query、Key 和 Value，代表“找什么”、“凭什么被选”和“汇聚什么内容”。失败于参数初始化造成秩崩溃。</td></tr>
<tr><td>注意力得分矩阵</td><td>计算 Q @ Kᵀ / √d_k。计算每一个 token 对所有其他 token 的关联概率分布。失败于长文本中 O(T²) 显存溢出。</td></tr>
<tr><td>注意力输出加权</td><td>将权重 softmax 之后，加权汇聚 Value 矩阵得到最终表征，输出 Z ∈ R^{B×T×d_model}。失败于缺少因果遮掩导致未来信息泄露。</td></tr>"""
    },
    "lec11": {
        "code": """# -*- coding: utf-8 -*-
# 第 11 讲: Transformer 与大模型基础 - PyTorch 搭建多头注意力机制 (Multi-Head Attention) 的自包含模块
import math
import torch
import torch.nn as nn

class MiniMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MiniMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 声明投影矩阵
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. 投影成 Q, K, V 并切分为 num_heads 个头
        # 形状变化: B x T x D -> B x T x h x d_k -> B x h x T x d_k
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算缩放点积注意力 (Scores 形状: B x h x T x T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. 应用因果遮掩 (Causal Masking) (防止自回归偷看未来)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. 汇聚 Value, 拼接多头输出 (B x h x T x d_k -> B x T x h x d_k -> B x T x D)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(context), attn_weights

# 测试
mha = MiniMultiHeadAttention(d_model=64, num_heads=4)
x = torch.randn(2, 5, 64) # Batch=2, Seq_len=5, Embedding=64

# 创建下三角因果 Mask
mask = torch.tril(torch.ones(5, 5)) # 5x5 下三角

output, weights = mha(x, mask=mask)
print("--- Mini Multi-Head Attention 测试 ---")
print(f"输入形状: {list(x.shape)}")
print(f"Causal Mask 矩阵:\\n{mask}")
print(f"输出形状: {list(output.shape)}") # 应为 [2, 5, 64]
print(f"注意力权重矩阵形状: {list(weights.shape)}") # 应为 [2, 4, 5, 5]
""",
        "flow_table": """<tr><td> 输入序列序列化</td><td>将输入句子切分为 Token 子词，经过 Embedding 并累加正弦/余弦位置编码 (Positional Encoding)。失败于输入序列长度超出上限。</td></tr>
<tr><td>多头投影分切</td><td>将 Token 特征多维投影，并切分成 h 个子特征空间，允许网络关注不同层次的语义空间。失败于维度不能被 num_heads 整除。</td></tr>
<tr><td>注意力缩放点积</td><td>计算 Q, K 投影相似度，处以 √d_k 进行数值稳定。自回归解码器在此处进行下三角 mask 掩码。失败于未合理 mask 导致预测发生作弊。</td></tr>
<tr><td>多头汇聚与连接</td><td>用 softmax 权重乘以 V 矩阵，再合并所有头的输出进行 W_o 线性投影。失败于残差连接未初始化造成梯度断裂。</td></tr>
<tr><td>FFN 前馈与规范化</td><td>进入双层全连接前馈层 (FFN)，各层块间使用 LayerNorm 和残差连接规范梯度流动。失败于 LayerNorm 导致深层表示坍缩。</td></tr>
<tr><td>自回归 Token 生成</td><td>在输出层通过 Linear + Softmax 计算下一个 token 概率分布，不断循环自回归迭代拼装生成语句。失败于陷入死循环重复生成。</td></tr>"""
    },
    "lec12": {
        "code": """# -*- coding: utf-8 -*-
# 第 12 讲: 目标检测与图像分割 - 计算检测框 Intersection over Union (IoU) 与非极大值抑制 (NMS) 的 NumPy 实现
import numpy as np

# 1. 计算两个框的交并比 (IoU)
# box 格式: [x1, y1, x2, y2]
def compute_iou(boxA, boxB):
    # 确定相交矩形的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # 计算相交面积
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # 计算各自面积
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # 计算并集面积
    unionArea = float(boxAArea + boxBArea - interArea)
    
    return interArea / unionArea if unionArea > 0 else 0.0

# 2. 非极大值抑制 (Non-Maximum Suppression, NMS)
# boxes: shape (N, 4), scores: shape (N,)
def nms(boxes, scores, iou_threshold=0.5):
    keep = []
    idxs = np.argsort(scores)[::-1] # 置信度从高到低排序
    
    while len(idxs) > 0:
        last = idxs[0]
        keep.append(last)
        
        # 计算当前高分框与其余所有框的 IoU
        ious = np.array([compute_iou(boxes[last], boxes[i]) for i in idxs[1:]])
        
        # 仅保留 IoU 小于阈值的框索引 (即剔除重合度高的重叠框)
        idxs = idxs[1:][ious < iou_threshold]
        
    return keep

# 测试数据
test_boxes = np.array([
    [100, 100, 210, 210], # 候选框 A
    [105, 105, 205, 205], # 候选框 B (与 A 高度重叠)
    [300, 320, 400, 420]  # 候选框 C (独立区域)
])
test_scores = np.array([0.9, 0.75, 0.82])

keep_indices = nms(test_boxes, test_scores, iou_threshold=0.5)
print("--- 目标检测 NMS 计算 ---")
print(f"输入候选框数: {len(test_boxes)}")
print(f"NMS (阈值 0.5) 保留框索引: {keep_indices}")
print(f"保留框坐标:\\n{test_boxes[keep_indices]}")
""",
        "flow_table": """<tr><td>图像提取特征</td><td>图像输入 CNN/ViT 特征网络得到高层语义特征图。保留微小边缘与局部拓扑特征。失败于感受野不足漏检大目标。</td></tr>
<tr><td>框候选或网格预测</td><td>使用区域生成网络 (RPN) 或密集多网格回归 (如 YOLO) 产生大量候选 Anchor 框及分类置信度。失败于背景物体极度不对称产生海量假阳性。</td></tr>
<tr><td>局部特征匹配</td><td>对候选框进行 RoI Align 或多网格映射，提取等尺寸局部区域的高维特征图。失败于位置量化误差导致定位漂移。</td></tr>
<tr><td>分类回归联合输出</td><td>多头任务分别预测 bounding box 的 Δx, Δy, Δw, Δh 偏移量以及各类目概率值。失败于回归损失和分类损失梯度失衡。</td></tr>
<tr><td>非极大值抑制 NMS</td><td>根据置信度分数排序，循环计算 IoU 交并比并过滤高度重叠框。输出精简检测框。失败于阈值太低导致同类近邻目标被错误擦除。</td></tr>
<tr><td>多指标 mAP 评估</td><td>在各个 IoU 阈值区间上计算精确度-召回率曲线下面积，输出最终 mAP@0.5/mAP@0.5:0.95。失败于误检样本数拉偏整体精度分。</td></tr>"""
    },
    "lec13": {
        "code": """# -*- coding: utf-8 -*-
# 第 13 讲: 强化学习与前沿模型 - PyTorch 实现变分自编码器 (VAE) 的重参数化与损失函数的极简训练步
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniVAE(nn.Module):
    def __init__(self, input_dim=8, latent_dim=2):
        super(MiniVAE, self).__init__()
        # Encoder: 预测均值与对数方差
        self.fc_enc = nn.Linear(input_dim, 16)
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        # Decoder: 从隐变量重建原始输入
        self.fc_dec1 = nn.Linear(latent_dim, 16)
        self.fc_dec2 = nn.Linear(16, input_dim)

    # 1. 重参数化技巧 (Reparameterization Trick)
    # z = mu + sigma * epsilon
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 采样标准正态噪声
        return mu + eps * std

    def forward(self, x):
        h = F.relu(self.fc_enc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_x = torch.sigmoid(self.fc_dec2(F.relu(self.fc_dec1(z))))
        return recon_x, mu, logvar

# 2. VAE 损失函数 = 重建损失 (MSE) + KL 散度约束
# Loss = E_q[log p(x|z)] - KL(q(z|x) || p(z))
def vae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL 散度闭式解: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# 演示前向计算与 Loss 梯度回传
model = MiniVAE()
x_toy = torch.rand(3, 8) # Batch=3, Feature=8

recon, mu, logvar = model(x_toy)
loss = vae_loss_fn(recon, x_toy, mu, logvar)

print("--- Mini VAE 训练前向计算 ---")
print(f"输入数据形状    : {list(x_toy.shape)}")
print(f"均值 mu 形状     : {list(mu.shape)} | 对数方差 logvar 形状: {list(logvar.shape)}")
print(f"重建输出形状    : {list(recon.shape)}")
print(f"总变分损失 loss : {loss.item():.4f}")
""",
        "flow_table": """<tr><td>真实数据分布</td><td>从高维真实像素或状态序列提取样本 x ~ p_data(x)。定义数据支持区域。失败于数据量极度匮乏。</td></tr>
<tr><td>潜变量 z 或噪声 ε</td><td>对 VAE 采样潜特征云层，对 Diffusion 注入噪声因子，或在 MDP 环境中采样状态转移。失败于采样方差过大。</td></tr>
<tr><td>生成器/解码器/去噪网络</td><td>使用神经网络将随机变量映射回数据像素空间。构建生成闭环。失败于模式崩溃 (Mode Collapse) 导致生成同质化。</td></tr>
<tr><td>判别或重构目标</td><td>对抗损失 (GAN)、重构与 KL 对偶 (VAE)、噪声拟合 (DDPM) 或 Bellman 目标 (RL)。失败于目标函数梯度不饱和或发散。</td></tr>
<tr><td>采样生成</td><td>通过交替优化或逐步反向去噪 (LDM) 获得新样本。智能体输出当前策略动作。失败于去噪累积误差偏离流形。</td></tr>
<tr><td>评价与失败分析</td><td>计算 FID 评估图像生成质量，计算奖励均值评估 RL 策略，根据自查分析优化方向。失败于指标虚高而主观表现恶劣。</td></tr>"""
    }
}

# Traverse notes HTML files
for lec_id, content in lecture_enrichments.items():
    html_path = f"{lec_id}/{lec_id}_notes.html"
    if not os.path.exists(html_path):
        print(f"File not found: {html_path}")
        continue
    
    print(f"Processing: {html_path}")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 1. Replace the code block under <section id="code">
    # Search for <section id="code">...<code>...</code></pre></section>
    # Note: Because the HTML is minified or on one/two lines, we use regex with DOTALL.
    # The code tag is usually <section id="code"><div class="section-title"><span class="num">09</span><h2>代码锚点：概念落到实现</h2></div><pre><code>...</code></pre>
    # Let's target: (<section id="code">.*?<code>)(.*?)(</code></pre>)
    code_pattern = re.compile(r'(<section id="code">.*?<code>)(.*?)(</code></pre>)', re.DOTALL)
    
    # Escape HTML characters in python code for embedding
    code_escaped = content["code"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    def repl_code(match):
        return match.group(1) + "\\n" + code_escaped + "\\n" + match.group(3)
        
    new_html = code_pattern.sub(repl_code, html_content)
    
    # 2. Replace the placeholder rows inside the flow table
    # Search for <tbody>...</tbody> inside <section id="flow">
    # The default table body has rows like: <tr><td>现实任务</td><td>输入是什么？输出是什么？...</td></tr>
    # Let's locate the table in <section id="flow">
    # We can match: (<section id="flow">.*?<tbody>)(.*?)(</tbody>.*?</table>)
    flow_pattern = re.compile(r'(<section id="flow">.*?<tbody>)(.*?)(</tbody>.*?</table>)', re.DOTALL)
    
    def repl_flow(match):
        return match.group(1) + content["flow_table"] + match.group(3)
        
    new_html = flow_pattern.sub(repl_flow, new_html)

    # 3. For lec13_notes.html, let's also fix the mixed up objectives and knowledge map to match GAN, VAE, Diffusion!
    if lec_id == "lec13":
        # Let's replace the Reinforcement Learning objectives under <section id="objectives">
        # with actual Generative Model objectives.
        obj_pattern = re.compile(r'(<section id="objectives">.*?<ul>)(.*?)(</ul>)', re.DOTALL)
        generative_objectives = """<li>理解生成对抗网络 (GAN) 的极小极大博弈公式及其交替优化算法。</li>
<li>掌握 Wasserstein GAN (WGAN) 用推土机距离解决传统 GAN 梯度消失和模式崩溃的数学原理。</li>
<li>理解变分自编码器 (VAE) 的重参数化技巧及其证据下界 (ELBO) 的推导过程。</li>
<li>掌握扩散模型 (DDPM) 的加噪与去噪数学公式，以及 Stable Diffusion 如何在潜空间降低计算成本。</li>
<li>理解对比学习机制 (CLIP, SigLIP) 与多模态桥接 (BLIP-2 Q-Former) 的先锋架构设计。</li>"""
        new_html = obj_pattern.sub(lambda m: m.group(1) + generative_objectives + m.group(3), new_html)

        # Let's replace the Reinforcement Learning map under <section id="map">
        map_pattern = re.compile(r'(<section id="map">.*?<div class="grid">)(.*?)(</div></section>)', re.DOTALL)
        generative_map = """<div class="card"><h3>GAN & WGAN</h3><p>极小极大博弈优化对抗损失，WGAN 引入 Wasserstein 距离与 1-Lipschitz 连续性保证梯度平滑。</p></div>
<div class="card"><h3>VAE</h3><p>引入隐变量估计，利用重参数化技巧解决采样不可微问题，最小化 ELBO 损失。</p></div>
<div class="card"><h3>Diffusion & SD</h3><p>马尔可夫链加噪，U-Net 拟合噪声去噪。Stable Diffusion 通过 VAE 压缩至 Latent 空间提速。</p></div>
<div class="card"><h3>多模态基座</h3><p>CLIP 跨模态对比学习对齐特征，BLIP-2 通过 Q-Former 桥接视觉特征与冻结的 LLM 大脑。</p></div>"""
        new_html = map_pattern.sub(lambda m: m.group(1) + generative_map + m.group(3), new_html)

    # Write back the enriched content
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(new_html)
    print(f"Successfully enriched {html_path}")

print("All notes enriched successfully!")
