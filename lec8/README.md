# 第 8 讲：深度学习（二）：卷积神经网络 CNN

> 课程：Introduction to Artificial Intelligence · 复旦大学  
> 日期：2026-04-29  
> 资料：人工智能-8-卷积神经网络.pdf、本讲 Markdown/实践材料

## 本讲定位

围绕图像输入的结构特性，系统讲解卷积、通道、padding、stride、池化、感受野、FLOPs 与经典 CNN。

## 学习目标

- 解释 CNN 为什么比全连接网络更适合图像：局部感知、参数共享、平移等变性。
- 能计算卷积输出尺寸、参数量、FLOPs 和感受野。
- 会用 PyTorch 写出标准 CNN，并理解 `Conv2d` 的输入输出维度。

## 知识地图

| 模块 | 内容 |
| --- | --- |
| 动机 | 1000×1000×3 图像若直接全连接会参数爆炸，且无法利用空间局部结构。 |
| 卷积层 | filter 在 H×W 上滑动，跨 channel 做加权求和，输出 feature map。 |
| 网络组件 | Padding 保边界，Stride/Pooling 下采样，GAP 降低分类头参数。 |
| 工程指标 | Params 影响模型大小，FLOPs 影响推理计算量；二者不能混为一谈。 |

## 核心概念

- `in_channels` 由输入决定，`out_channels` 由设计者设定，kernel/stride/padding 控制空间尺寸。
- Depthwise convolution 对每个通道单独卷积，配合 1×1 pointwise convolution 构成轻量 CNN。
- CNN 从边缘纹理到部件再到语义对象逐层抽象，是目标检测、分割、生成模型的视觉基础。

## 课堂讲解补充

CNN 是本课程计算机视觉能力链的核心。卷积把图像的空间局部性和参数共享写进网络结构，使模型能高效学习边缘、纹理、部件和对象语义。真正掌握 CNN，需要同时会讲直觉、会算尺寸/参数/FLOPs、会写 PyTorch `Conv2d` 代码，并能解释它如何支撑检测、分割和生成模型。

### 复习组织方式

- **问题背景**：这节课为什么要引入这个概念。
- **方法主线**：算法或模型按什么步骤工作。
- **公式/代码**：至少抓住一个能落地计算的表达。
- **局限性**：说明它在哪些场景下会失效或需要改进。

## 公式与记忆点

- 输出尺寸：H_out=⌊(H+2P-K)/S⌋+1。
- 参数量：C_out × (C_in × K_h × K_w + bias)。
- FLOPs 近似：2 × C_out × C_in × K_h × K_w × H_out × W_out。
- 感受野递推：RF_new = RF_old + (K-1) × 累计 stride。

## 易错点

- 把 batch 维和 channel 维写反：PyTorch 图像输入必须是 `[N,C,H,W]`。
- 只看参数量不看 FLOPs，会误判模型推理速度。
- padding 与 stride 选择不当会让特征图尺寸提前坍缩。

## 课后巩固

- 计算 `Conv2d(3,64,3,padding=1)` 在 224×224 输入上的输出尺寸和参数量。
- 把上面的 SmallCNN 改成 depthwise separable convolution 版本。

## 文件索引

| 文件 | 说明 |
|---|---|
| `lec8_notes.html` | 详细课程笔记网页 |
| `lec8.md` | 本讲补充材料 |

## 关键词

`CNN`, `Convolution`, `Padding/Stride`, `Pooling`, `Receptive Field`, `FLOPs`, `CV`
