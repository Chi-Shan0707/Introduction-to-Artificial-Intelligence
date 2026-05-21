# 第8讲：深度学习（二）—— 卷积神经网络（CNN）

> 丁恒辉，2026/4/23 | 基于 PDF 课件 + 课堂转录 + 补充整理

---

## 本讲概览

本讲围绕**卷积神经网络（CNN）**展开，从全连接网络处理图像的困境出发，引出卷积的两大核心设计——**局部感知**和**参数共享**，并系统讲解卷积操作的各个维度（通道、核大小、padding、stride）以及感受野、下采样、池化等关键概念。

此外，本讲还涉及**分类模型评估指标**与**CLIP Score**（图文对齐评估），以及 CNN 的**参数量（Params）**和**计算量（FLOPs）**的计算方法。

---

## 知识地图

```
CNN 核心
├── 为什么需要 CNN
│   ├── 全连接处理图像 → 参数爆炸（1000×1000×3 ≈ 3M 输入）
│   ├── 局部感知（每个神经元只看局部区域）
│   └── 参数共享（同一卷积核在整张图上滑动）
├── 卷积操作
│   ├── 基本原理（kernel 滑动 → 逐元素相乘求和）
│   ├── 多通道处理（每个 filter 的 channel 数 = in_channel）
│   ├── Depthwise Convolution（每个 filter 只跟单个 channel 相乘）
│   └── Group Convolution（将通道分组，group=1 即标准卷积）
├── 关键参数
│   ├── in_channel（由输入决定，不能自己写）
│   ├── out_channel / kernel_size / stride（自己定义）
│   ├── padding（一般与 kernel_size 保持对应关系）
│   └── 输出尺寸公式：⌊(Input + 2×Padding - Kernel) / Stride⌋ + 1
├── 感受野（Receptive Field）
│   ├── 单层 = kernel_size
│   └── 逐层增大：RF_new = RF_old + (K-1) × ∏Stride
├── 下采样（Downsampling）
│   ├── Stride > 1 的卷积
│   ├── Max Pooling / Average Pooling
│   └── Global Average Pooling（GAP）
├── Params 与 FLOPs
│   ├── Params = C_out × (C_in × K_h × K_w + 1)，与 H×W 无关
│   ├── FLOPs = 2 × C_out × C_in × K_h × K_w × H_out × W_out
│   └── Params 少 ≠ FLOPs 少
└── 评估指标
    ├── 分类：Accuracy / Precision / Recall / F1 / AUC-ROC / AUC-PR / Log Loss
    └── 多模态：CLIP Score（图文余弦相似度）
```

---

## 核心公式速查

| 主题 | 公式 |
|---|---|
| 输出尺寸 | $\lfloor (Input + 2 \times Padding - Kernel) / Stride \rfloor + 1$ |
| Same Padding | $Padding = (KernelSize - 1) / 2$ |
| 感受野 | $RF_{new} = RF_{old} + (K - 1) \times \prod Stride_i$ |
| 卷积参数量 | $Params = C_{out} \times (C_{in} \times K_h \times K_w + 1)$ |
| 卷积 FLOPs | $FLOPs = 2 \times C_{out} \times C_{in} \times K_h \times K_w \times H_{out} \times W_{out}$ |
| F1-Score | $F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ |
| CLIP Score | $\cos(I, T) = \frac{f_{img}(img) \cdot f_{txt}(txt)}{\|f_{img}(img)\| \|f_{txt}(txt)\|}$ |

---

## CNN 关键设计原则

| 设计 | 解决的问题 | 直觉 |
|---|---|---|
| **局部感知** | 全连接参数爆炸 | 每个神经元只关注局部区域，而非整张图 |
| **参数共享** | 参数量仍然太大 | 同一卷积核在图像各位置复用，利用平移不变性 |
| **多通道** | RGB 等多通道输入 | 每个 filter 的深度 = in_channel，堆叠多个 filter 得到多通道输出 |
| **Padding** | 边缘像素被"扫到"次数少 | 四周补 0，使边缘与中心享有同等关注 |
| **Stride > 1** | 特征图太大，计算量高 | 跳步采样，实现下采样 |
| **池化** | 需要进一步降维 + 增强平移不变性 | 局部区域取最大/平均，无参数 |

---

## 标准卷积 vs Depthwise 卷积

| 对比 | 标准卷积 | Depthwise 卷积 |
|---|---|---|
| filter 与通道关系 | 每个 filter 跨所有 channel | 每个 filter 只对应 1 个 channel |
| 参数量 | $C_{out} \times C_{in} \times K^2$ | $C_{in} \times K^2$（大幅减少） |
| PyTorch 参数 | `groups=1`（默认） | `groups=C_in` |
| 典型应用 | 通用特征提取 | MobileNet 等轻量网络 |

> Depthwise + Pointwise (1×1 Conv) = **深度可分离卷积**（Depthwise Separable Conv），大幅减少 Params 和 FLOPs。

---

## 评估指标选择速查

| 场景 | 推荐指标 |
|---|---|
| 样本均衡 | Accuracy + F1-Score |
| 样本不均衡（1:10） | F1-Score + AUC-ROC |
| 极度不均衡（1:1000+） | **PRAUC** + F1-Score |
| 需要概率输出 | Log Loss |
| 文生图 / 图像描述 | **CLIP Score** |

---

## 课堂原话摘录

> "in_channel 等于多少？你要看人家 X 来的等于多少，你不能瞎写。那 out_channel 呢是你自己可以去定义的。"
>
> "你的 padding 呢一般要跟你的 kernel size 要去保持一个对应的关系。"
>
> "depthwise convolution，每个 filter 我只跟它单独自己本身一个 channel 去相乘。放到 PyTorch 里面呢，它是有一个参数叫做 group。"

---

## 课后任务

- 下节课需自带电脑，提前准备好环境
- 完成两道卷积操作计算题（巩固输出尺寸、参数量、FLOPs 的计算）

---

## 文件索引

| 文件 | 内容 |
|---|---|
| `lec8.md` | 完整笔记：评估指标（1-10节）+ CNN 核心概念（11.1-11.9节） |
| `transcript.md` | 课堂转录（含元宝纪要） |
