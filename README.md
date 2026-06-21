# Introduction to Artificial Intelligence

复旦大学人工智能导论课程整理仓库。本文档按课程节次统一索引每讲 README 与可视化 HTML 笔记；每讲内容均以课件 PDF、已有 Markdown 和课程实践材料为依据重新组织。

## 课程主线

本课程从符号主义与搜索开始，逐步进入机器学习、深度学习、计算机视觉、自然语言处理、Transformer/大模型与强化学习。整理后的笔记采用统一结构：**课程定位 → 学习目标 → 知识地图 → 核心概念 → 公式/代码 → 易错点 → 课后巩固**。

重点能力链：

- **CV/CNN**：第 8 讲系统讲解卷积神经网络；第 12 讲扩展到检测、分割、视频分割与图像生成。
- **PyTorch 实践**：第 7 讲建立神经网络训练基础，第 9 讲落到 Tensor、Autograd、DataLoader 与图像分类训练循环。
- **大模型与多模态**：第 10-12 讲从 RNN/Attention 到 Transformer、LLM、CV&NLP 前沿。
- **强化学习项目**：第 13 讲连接 MDP、Bellman、Q-learning、策略梯度与 CartPole 项目。

## 课程目录

> GitHub Pages 首页已改为逐讲深度笔记地图；深度内容直接写入每一讲 `lec*_notes.html`，而不是单独集中到统一页面。

| 周次 | README | HTML 笔记 | 关键词 |
|---|---|---|---|
| 第 1 讲 | [绪论：人工智能是什么](lec1/README.md) | [HTML notes](lec1/lec1_notes.html) | AI定义, 课程考核, 发展阶段, CV研究案例 |
| 第 2 讲 | [逻辑与知识表示](lec2/README.md) | [HTML notes](lec2/lec2_notes.html) | 形式逻辑, 数理逻辑, 知识图谱, FOIL |
| 第 3 讲 | [搜索求解](lec3/README.md) | [HTML notes](lec3/lec3_notes.html) | 状态空间, BFS, DFS, UCS |
| 第 4 讲 | [机器学习（一）：线性模型](lec4/README.md) | [HTML notes](lec4/lec4_notes.html) | 监督学习, 线性回归, Logistic回归, 损失函数 |
| 第 5 讲 | [机器学习（二）：SVM 与模型选择评估](lec5/README.md) | [HTML notes](lec5/lec5_notes.html) | SVM, 最大间隔, 核函数, 交叉验证 |
| 第 6 讲 | [机器学习（三）：无监督学习](lec6/README.md) | [HTML notes](lec6/lec6_notes.html) | K-Means, PCA, GMM, EM |
| 第 7 讲 | [深度学习（一）：神经网络基础](lec7/README.md) | [HTML notes](lec7/lec7_notes.html) | MLP, Backprop, Activation, Regularization |
| 第 8 讲 | [深度学习（二）：卷积神经网络 CNN](lec8/README.md) | [HTML notes](lec8/lec8_notes.html) | CNN, Convolution, Padding/Stride, Pooling |
| 第 9 讲 | [深度学习（三）：PyTorch 实践](lec9/README.md) | [HTML notes](lec9/lec9_notes.html) | PyTorch, Tensor, Autograd, DataLoader |
| 第 10 讲 | [序列建模：RNN、LSTM 与注意力](lec10/README.md) | [HTML notes](lec10/lec10_notes.html) | RNN, LSTM, GRU, Attention |
| 第 11 讲 | [Transformer 与大模型基础](lec11/README.md) | [HTML notes](lec11/lec11_notes.html) | Self-Attention, Multi-Head, Positional Encoding, Transformer |
| 第 12 讲 | [计算机视觉与自然语言处理：检测、分割、生成](lec12/README.md) | [HTML notes](lec12/lec12_notes.html) | Object Detection, Segmentation, R-CNN, YOLO |
| 第 13 讲 | [强化学习：从 MDP 到深度强化学习](lec13/README.md) | [HTML notes](lec13/lec13_notes.html) | RL, MDP, Bellman, Q-Learning |

## 使用建议

1. 先读每讲 `README.md` 建立知识框架。
2. 再打开对应 `lec*_notes.html`，按网页中的流程学习细节、公式和代码。
3. 对 CV/CNN/PyTorch 相关内容，建议配合第 8、9、12 讲反复交叉复习：先算维度，再写 PyTorch 模型，再用检测/分割指标解释输出。
4. 汇报或考试复习时，统一使用“任务定义、核心思想、数学表达、实现细节、评价指标、局限性”的表达格式。

## 文件说明

- `lec*/README.md`：每讲速览、知识地图与复习清单。
- `lec*/lec*_notes.html`：统一视觉风格的详细可视化课程笔记。
- `lec*/lec*.md`：原始整理笔记或补充材料（部分讲次提供）。
- `lec9/course_materials/`：PyTorch/FashionMNIST/CIFAR 等实践材料（若已上传）。
- `pj12_cartpole/`：强化学习 CartPole 项目材料（若已上传）。
