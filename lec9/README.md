# 第 9 讲：深度学习（三）：PyTorch 实践

> 课程：Introduction to Artificial Intelligence · 复旦大学  
> 日期：2026-05-07  
> 资料：无 PDF；基于 course_materials notebooks 与实践材料

## 本讲定位

把神经网络落到可运行代码：Tensor、Autograd、Dataset/DataLoader、训练循环、FashionMNIST 与 CIFAR 示例。

## 学习目标

- 掌握 PyTorch 张量、自动微分和 `nn.Module` 的最小工作流。
- 能写出标准训练循环：前向、损失、清梯度、反传、优化、评估。
- 理解图像预处理、数据增强和 CNN 训练在 CV 任务中的作用。

## 知识地图

| 模块 | 内容 |
| --- | --- |
| 张量基础 | 创建、索引、广播、设备迁移、dtype/shape 检查。 |
| 自动微分 | 计算图记录操作，`loss.backward()` 自动填充梯度。 |
| 数据管线 | Dataset 定义样本，DataLoader 负责 batch、shuffle 和并行读取。 |
| 实践任务 | FashionMNIST、CIFAR100、图像预处理 notebook 构成完整 CV 入门实验。 |

## 核心概念

- PyTorch 的优势是动态图：前向计算即构图，调试体验接近普通 Python。
- 训练模式 `model.train()` 与评估模式 `model.eval()` 会影响 Dropout/BatchNorm。
- CV 代码要始终检查 image tensor 的尺度、归一化方式和类别映射。

## 课堂讲解补充

PyTorch 实践把前面所有深度学习概念变成可运行实验。Tensor 是数据容器，Autograd 是梯度引擎，`nn.Module` 是模型组织方式，DataLoader 是训练数据入口。FashionMNIST/CIFAR 实验的目标不是追求最高精度，而是建立标准训练循环和调试习惯。

### 复习组织方式

- **问题背景**：这节课为什么要引入这个概念。
- **方法主线**：算法或模型按什么步骤工作。
- **公式/代码**：至少抓住一个能落地计算的表达。
- **局限性**：说明它在哪些场景下会失效或需要改进。

## 公式与记忆点

- 交叉熵：L=-Σ y_c log p_c。
- 梯度更新：θ ← optimizer(θ, ∇θL)。
- 图像标准化：x_norm=(x-mean)/std。

## 易错点

- 忘记把数据和模型放到同一 device 会报错。
- 评估时不使用 `torch.no_grad()` 会浪费显存。
- 训练 loss 正常但 accuracy 极低时，优先检查标签、归一化和输出类别数。

## 课后巩固

- 运行 `course_materials/handson_fashionmnist.ipynb`，记录训练/验证准确率。
- 基于 CIFAR100 notebook 替换模型为 SmallCNN，并比较性能。

## 文件索引

| 文件 | 说明 |
|---|---|
| `lec9_notes.html` | 详细课程笔记网页 |
| `helloworld.sh` | 本讲补充材料 |
| `course_materials/` | PyTorch 实践 notebook、示例图片与参考文件 |

## 关键词

`PyTorch`, `Tensor`, `Autograd`, `DataLoader`, `FashionMNIST`, `CIFAR`, `训练循环`
