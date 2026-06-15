# 常见分类模型评估指标与打分方式

## 1. 混淆矩阵（Confusion Matrix）—— 一切的基础

以二分类为例：

|                | 预测 Positive | 预测 Negative |
|----------------|--------------|--------------|
| 实际 Positive  | TP（真阳性） | FN（假阴性） |
| 实际 Negative  | FP（假阳性） | TN（真阴性） |

---

## 2. Accuracy（准确率）

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

- **含义**：所有样本中预测正确的比例。
- **适用**：正负样本均衡时效果好；**样本不均衡时会有误导**（例如 99% 正样本，全猜正也有 99% 准确率）。

---

## 3. Precision（精确率）& Recall（召回率）

$$Precision = \frac{TP}{TP + FP}$$

$$Recall = \frac{TP}{TP + FN}$$

- **Precision**：预测为正的样本中，真正为正的比例 → **"查准率"**
- **Recall**：实际为正的样本中，被正确找出来的比例 → **"查全率"**
- 二者通常是 **trade-off** 关系：提高阈值 → Precision↑ Recall↓；降低阈值 → Precision↓ Recall↑

---

## 4. F1-Score（F1 分数）

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} = \frac{2TP}{2TP + FP + FN}$$

- **含义**：Precision 和 Recall 的 **调和平均数**，兼顾两者。
- **特点**：对正负样本不均衡的场景比 Accuracy 更可靠。
- **推广**：$F_\beta = (1+\beta^2) \cdot \frac{Precision \cdot Recall}{\beta^2 \cdot Precision + Recall}$
  - $\beta > 1$：更侧重 Recall（如疾病筛查）
  - $\beta < 1$：更侧重 Precision（如垃圾邮件过滤）
  - $\beta = 1$：即 F1

---

## 5. ROC 曲线与 AUC-ROC

### ROC 曲线
- 横轴：**FPR**（False Positive Rate）$= \frac{FP}{FP + TN}$
- 纵轴：**TPR**（True Positive Rate）$= \frac{Recall} = \frac{TP}{TP + FN}$
- 通过不断改变分类阈值，绘制 (FPR, TPR) 点，连成曲线。

### AUC-ROC
- ROC 曲线下面积，取值 $[0, 1]$。
- **AUC = 1**：完美分类器；**AUC = 0.5**：随机猜测。
- **含义**：随机取一个正样本和一个负样本，模型将正样本排在负样本前面的概率。
- **优点**：对正负样本比例不敏感，适合评估排序能力。

---

## 6. PR 曲线与 AUC-PR（PRAUC）

### PR 曲线
- 横轴：**Recall**，纵轴：**Precision**。
- 通过不断改变阈值，绘制 (Recall, Precision) 点，连成曲线。

### AUC-PR（PRAUC）
- PR 曲线下面积。
- **相比 AUC-ROC 的优势**：在 **正负样本极度不均衡** 时（如正样本只占 0.1%），PRAUC 更有区分度。
  - ROC 在不均衡时可能看起来很好（因为 TN 很多，FPR 看起来低），但 PR 曲线能暴露 Precision 的问题。
- **适用场景**：信息检索、推荐系统、欺诈检测等不均衡场景。

### PRAUC 计算方法（代码层面）
```python
# sklearn 方式
from sklearn.metrics import average_precision_score
ap = average_precision_score(y_true, y_scores)  # PRAUC 近似值
```
> 注意：`average_precision_score` 计算的是 **Average Precision (AP)**，是 PRAUC 的一个保守近似（矩形插值），与逐点梯形插值的 PRAUC 略有差异。

---

## 7. Log Loss（对数损失 / 交叉熵）

$$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$

- $y_i$：真实标签（0 或 1），$p_i$：预测为正的概率。
- **含义**：衡量预测概率与真实标签的差距，输出的是概率而非硬分类。
- **特点**：**惩罚自信的错误预测**（预测 0.99 但实际为 0 → 损失极大）。
- **取值**：$[0, +\infty)$，越小越好。

---

## 8. CLIP Score（图文对齐分数）

### 背景
CLIP（Contrastive Language-Image Pre-training）是 OpenAI 提出的视觉-语言预训练模型，能将图像和文本映射到同一语义空间。**CLIP Score 利用这一特性来衡量图文之间的语义一致性**。

### 计算方法

$$CLIP\text{-}Score = \cos(I, T) = \frac{I \cdot T}{\|I\| \|T\|}$$

其中：
- $I = f_{image}(image)$：图像经 CLIP Image Encoder 得到的 embedding 向量
- $T = f_{text}(text)$：文本经 CLIP Text Encoder 得到的 embedding 向量
- $\cos(\cdot)$：余弦相似度

**步骤**：
1. 用 CLIP 的 Image Encoder 提取图像特征 $I \in \mathbb{R}^{512}$（ViT-L/14）
2. 用 CLIP 的 Text Encoder 提取文本特征 $T \in \mathbb{R}^{512}$（Transformer）
3. 计算 $I$ 和 $T$ 的余弦相似度，即为 CLIP Score

### 取值与含义
- **取值范围**：$[0, 1]$（经过 sigmoid 归一化后），越大越好
- **未归一化**的余弦相似度范围约为 $[-1, 1]$

### 代码示例
```python
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

inputs = processor(text=["a dog sitting on the grass"], images=dog_image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# CLIP Score = image-text cosine similarity
clip_score = outputs.logits_per_image.softmax(dim=1)[0, 0].item()
print(f"CLIP Score: {clip_score:.4f}")
```

### 优缺点
- **优点**：无需人工标注，自动评估图文对齐质量；语义理解能力强
- **缺点**：CLIP 本身有偏好偏差（如偏好英文、偏好写实风格）；**不能替代人类主观评价**；对细节（文字渲染、手指数量）不敏感
- **变体**：**CLIP Score (Ref)**（使用参考图像的改进版，比原始 CLIP Score 与人类判断相关性更高）

### 应用场景
- **文生图**（Text-to-Image）：评估生成图像与输入 prompt 的匹配度
- **图像描述**（Image Captioning）：评估生成的 caption 与图像的一致性
- **图像编辑**：评估编辑后图像与目标描述的对齐程度

### 与其他图像质量指标的对比

| 指标 | 评估维度 | 是否需要参考图 | 依赖预训练模型 |
|------|---------|--------------|--------------|
| CLIP Score | 图文语义对齐 | 否（只用 prompt） | 是（CLIP） |
| FID | 图像分布质量 | 是（真实图像集） | 是（Inception） |
| IS（Inception Score） | 生成图像质量 & 多样性 | 否 | 是（Inception） |
| LPIPS | 感知相似度 | 是（参考图像） | 是（VGG/AlexNet） |
| 人类评价 | 整体质量 | 通常否 | 否 |

> **核心区别**：CLIP Score 是 **无参考（reference-free）** 指标——不需要真实参考图，只需文本描述就能评估，这是它最大的优势。

---

## 9. 各指标对比总结

| 指标 | 关注点 | 适合场景 | 不均衡数据 |
|------|--------|---------|-----------|
| Accuracy | 整体正确率 | 均衡分类 | 不推荐 |
| Precision | 减少误报 | 垃圾邮件、推荐 | 一般 |
| Recall | 减少漏报 | 疾病筛查、安全 | 一般 |
| F1-Score | 兼顾 P 和 R | 通用不均衡场景 | 推荐 |
| AUC-ROC | 排序能力 | 通用评估 | 较好 |
| **AUC-PR (PRAUC)** | 精确率-召回率权衡 | **极度不均衡** | **最推荐** |
| LogLoss | 概率校准 | 需要概率输出的场景 | 较好 |
| CLIP Score | 图文语义对齐 | 文生图/图像描述 | N/A（多模态） |

---

## 10. 实战选择建议

1. **样本均衡** → Accuracy + F1-Score
2. **样本不均衡**（如 1:10）→ F1-Score + AUC-ROC
3. **极度不均衡**（如 1:1000+）→ **PRAUC** + F1-Score
4. **需要概率输出** → LogLoss
5. **关注排序质量**（如推荐系统）→ **PRAUC** 或 NDCG（检索场景用 MAP/NDCG）

---

## 11. CNN 核心概念与输出尺寸计算（期末必考）

### 11.1 Padding（填充）

**为什么需要 Padding？**

卷积核在图像上滑动时，边缘像素被卷积核"扫到"的次数比中心像素**少**——边缘信息被浪费了。Padding 就是在图片**四周（上下左右）填充额外的像素**（通常填 0），使得：

1. **边缘像素也能被完整覆盖**，和中心像素享有同等的"被关注"机会
2. **可以控制输出尺寸**，使其保持与输入相同

```
无 Padding（Padding=0）：          有 Padding（Padding=1）：

  ■ ■ ■ ■ ■                      0 0 0 0 0 0 0
  ■ ■ ■ ■ ■                      0 ■ ■ ■ ■ ■ 0
  ■ ■ ■ ■ ■   5×5 输入            0 ■ ■ ■ ■ ■ 0   填充后 7×7
  ■ ■ ■ ■ ■   3×3 卷积核          0 ■ ■ ■ ■ ■ 0   再用 3×3 卷积
  ■ ■ ■ ■ ■   输出 3×3 ← 缩小了   0 ■ ■ ■ ■ ■ 0   输出 5×5 = 原尺寸
                                   0 0 0 0 0 0 0
```

**三种 Padding 情况：**

| Padding | 效果 | 输出尺寸 |
|---------|------|---------|
| **Padding=0**（Valid） | 不填充，边缘丢失，尺寸缩小 | $Input - Kernel + 1$ |
| **Padding=(K-1)/2**（Same） | 填充使输出=输入 | $Input$（尺寸不变） |
| **Padding > (K-1)/2** | 过度填充，尺寸反而变大 | $> Input$ |

**Padding 填充在图片两端（上下左右各填 Padding 个像素）**，所以公式里是 $2 \times Padding$。

---

### 11.2 How to Keep the Same Output Size as Input

要让卷积后的输出尺寸与输入相同（即 "Same Padding"），需要：

$$Padding = \frac{KernelSize - 1}{2}$$

**推导**：令 OutputSize = InputSize，代入公式：

$$Input = \frac{Input + 2 \times Padding - Kernel}{Stride} + 1$$

当 Stride = 1 时：

$$Input = Input + 2 \times Padding - Kernel + 1$$

$$2 \times Padding = Kernel - 1 \implies Padding = \frac{Kernel - 1}{2}$$

**常见情况：**
- Kernel=3 → Padding=1
- Kernel=5 → Padding=2
- Kernel=7 → Padding=3

> **必记结论：奇数核 + Stride=1 + Padding=(K-1)/2 → 输出尺寸不变**

---

### 11.3 输出尺寸通用公式

$$OutputSize = \lfloor \frac{InputSize + 2 \times Padding - KernelSize}{Stride} \rfloor + 1$$

| 参数 | 含义 | 典型值 |
|------|------|--------|
| **InputSize** | 输入特征图的宽/高 | 如 32×32 |
| **KernelSize** | 卷积核大小 | 3×3, 5×5, 7×7 |
| **Padding** | 边界填充的像素数 | 0（无填充）, 1（3×3核保持尺寸不变） |
| **Stride** | 卷积核每次移动的步长 | 通常为 1 或 2 |

### 计算示例

**示例 1**：输入 32×32，Kernel=3，Padding=1，Stride=1（Same Padding）
$$Output = \lfloor \frac{32 + 2 \times 1 - 3}{1} \rfloor + 1 = \lfloor 31 \rfloor + 1 = 32$$
→ 尺寸不变

**示例 2**：输入 32×32，Kernel=5，Padding=0，Stride=1（Valid Padding）
$$Output = \lfloor \frac{32 + 0 - 5}{1} \rfloor + 1 = 27 + 1 = 28$$
→ 尺寸缩小

**示例 3**：输入 28×28，Kernel=2，Padding=0，Stride=2（池化层）
$$Output = \lfloor \frac{28 - 2}{2} \rfloor + 1 = 13 + 1 = 14$$
→ 尺寸减半

### 注意事项
- $\lfloor \cdot \rfloor$ 是**向下取整**，当分母不能整除时尺寸会丢掉边缘像素
- PyTorch 中设 `padding="same"` 可自动计算，无需手算

---

### 11.4 感受野（Receptive Field）

**定义**：输出特征图上某个像素对应到输入图像上的区域大小。

**直觉**：一个神经元的"视野范围"——它能"看到"原始图像的多少像素。

```
输入层 (5×5)          第1层卷积 (3×3)        第2层卷积 (3×3)

┌─────────────┐      ┌───────────┐        ┌─────┐
│ ■ ■ ■ ■ ■   │      │ A B C     │        │ X   │
│ ■ ■ ■ ■ ■   │ ──→  │ D E F     │ ──→   └─────┘
│ ■ ■ ■ ■ ■   │      │ G H I     │
│ ■ ■ ■ ■ ■   │
│ ■ ■ ■ ■ ■   │

第2层的 X 对应输入层的 5×5 区域 → 感受野 = 5×5
第1层的 E 对应输入层的 3×3 区域 → 感受野 = 3×3
```

**感受野逐层增大**：网络越深，每个神经元"看到"的原始图像区域越大。

**感受野计算公式**（两层叠加为例）：

$$RF_{new} = RF_{old} + (KernelSize - 1) \times \prod_{i} Stride_i$$

- 第1层：Kernel=3, Stride=1 → RF = 3
- 第2层：Kernel=3, Stride=1 → RF = 3 + (3-1)×1 = **5**
- 第3层：Kernel=3, Stride=1 → RF = 5 + (3-1)×1 = **7**
- 第4层：Kernel=3, Stride=2 → RF = 7 + (3-1)×2 = **11**

> **为什么感受野重要？** 感受野越大，模型能捕捉的上下文信息越多。浅层感受野小 → 捕捉局部纹理/边缘；深层感受野大 → 捕捉全局语义/形状。

---

### 11.5 下采样（Downsampling）

**定义**：降低特征图空间分辨率（宽高变小）的过程。

**为什么需要下采样？**
1. **减少计算量**：尺寸变小 → 参数量和计算量大幅降低
2. **扩大感受野**：不增加网络深度的前提下，让后续层看到更大的区域
3. **增加平移不变性**：小范围的位置变化不影响特征表达

**实现方式：**

| 方式 | 说明 | 是否可学习 |
|------|------|-----------|
| **Stride > 1 的卷积** | 卷积步长为 2，直接跳着采样 | 是（卷积参数可学习） |
| **池化（Pooling）** | 在局部区域内取最大值或平均值 | 否（固定操作） |
| **全局平均池化（GAP）** | 每个通道取所有像素的平均值，输出 1×1 | 否 |

---

### 11.6 池化（Pooling）

**作用**：对局部区域进行聚合，实现下采样，增强平移不变性。

#### 最大池化（Max Pooling）——最常用

在局部窗口内取**最大值**。

```
输入 4×4：                Max Pooling 2×2, Stride=2：

1  3  2  1                3  4
4  6  3  2       →        8  6
2  8  5  1               输出 2×2
3  4  1  0
```

- **保留最显著特征**（最强的激活值）
- **没有可学习的参数**——只是取最大值
- **输出尺寸**同样遵循通用公式

#### 平均池化（Average Pooling）

在局部窗口内取**平均值**。

```
输入 4×4：                Avg Pooling 2×2, Stride=2：

1  3  2  1                3.5  2.0
4  6  3  2       →        4.25 1.75
2  8  5  1               输出 2×2
3  4  1  0
```

- **保留整体信息**，特征更平滑
- 常用于网络末端（如 ResNet 最后的 Global Average Pooling）

#### 池化 vs Stride 卷积

| 对比 | Max/Avg Pooling | Stride-2 卷积 |
|------|----------------|---------------|
| 可学习参数 | 无 | 有 |
| 信息保留 | Pooling 丢弃部分信息 | 卷积保留更多信息 |
| 计算量 | 更小 | 更大 |
| 现代趋势 | 仍广泛使用 | ResNet 等用 Stride 卷积替代部分 Pooling |

> **关键点**：池化层**只改变空间尺寸（H×W）**，**不改变通道数（C）**。改变通道数的是卷积层（由卷积核的数量决定）。

---

### 11.7 CNN 参数量（Params）

**定义**：模型中所有可学习权重（weights）和偏置（bias）的总数，决定了模型的**存储大小**。

#### 单层卷积的参数量

$$Params_{conv} = C_{out} \times (C_{in} \times K_h \times K_w + 1_{bias})$$

其中：
- $C_{in}$：输入通道数
- $C_{out}$：输出通道数（= 卷积核数量）
- $K_h \times K_w$：卷积核大小（如 3×3）
- $+1$：每个输出通道一个 bias（可省略设 bias=False）

**直观理解**：每个输出通道需要 $C_{in} \times K_h \times K_w$ 个权重来"汇总"所有输入通道的信息，共 $C_{out}$ 个输出通道。

#### 计算示例

**示例 1**：输入 3 通道（RGB），输出 64 通道，Kernel=3×3，带 bias
$$Params = 64 \times (3 \times 3 \times 3 + 1) = 64 \times 28 = 1{,}792$$

**示例 2**：输入 64 通道，输出 128 通道，Kernel=3×3，无 bias
$$Params = 128 \times (64 \times 3 \times 3) = 128 \times 576 = 73{,}728$$

> **注意**：卷积的参数量与输入图像的空间尺寸（H×W）**无关**！只与通道数和核大小有关。

#### 全连接层（Linear）的参数量

$$Params_{fc} = C_{in} \times C_{out} + C_{out}$$

全连接层的参数量通常**远大于**卷积层，这也是 CNN 比全连接网络参数少的核心原因——**权值共享**。

#### 经典网络参数量对比

| 网络 | Params (M) | 特点 |
|------|-----------|------|
| LeNet-5 | ~0.06 | 最早，极轻量 |
| AlexNet | ~60 | 首次用 ReLU、Dropout |
| VGG-16 | ~138 | 3×3 小核堆叠，参数量大 |
| GoogLeNet | ~6.8 | Inception 模块，大幅减少参数 |
| ResNet-18 | ~11.7 | 残差连接 |
| ResNet-50 | ~25.6 | 最常用的 baseline |

> **考试重点**：VGG 为什么参数多？大量 3×3 卷积堆叠，且通道数较大（512, 512）。GoogLeNet 如何减少参数？用 1×1 卷积降维（bottleneck）。

---

### 11.8 FLOPs（浮点运算次数）

**定义**：FLoating-point OPerations，即模型进行一次前向推理所需的**浮点运算总次数**，决定了模型的**计算开销/推理速度**。

> **注意区分**：
> - **FLOPs**（小写 s）= 浮点运算次数（如 1 GFLOPs = 10^9 次运算）
> - **FLOPS**（大写 S）= 每秒浮点运算次数（衡量硬件速度，如 GPU 算力）

#### 单层卷积的 FLOPs

$$FLOPs_{conv} = 2 \times C_{out} \times C_{in} \times K_h \times K_w \times H_{out} \times W_{out}$$

其中：
- 系数 **2** = 1 次乘法 + 1 次加法（一次 MAC = 2 FLOPs）
- $H_{out} \times W_{out}$：输出特征图的空间尺寸

**与参数量的关键区别**：参数量只与 $C_{in}, C_{out}, K$ 有关；**FLOPs 还与空间尺寸 $H \times W$ 有关**。

#### 计算示例

**示例**：输入 64 通道 28×28，输出 128 通道，Kernel=3×3，Padding=1，Stride=1
- 输出尺寸 = 28×28
$$FLOPs = 2 \times 128 \times 64 \times 3 \times 3 \times 28 \times 28 = 109{,}890{,}048 \approx 0.11 \text{ GFLOPs}$$

#### 各层 FLOPs 对比

| 层类型 | FLOPs 公式 | 备注 |
|--------|-----------|------|
| 卷积层 | $2 \times C_{out} \times C_{in} \times K \times H_{out} \times W_{out}$ | 计算最密集的层 |
| 全连接层 | $2 \times C_{in} \times C_{out}$ | 参数多但通常只出现在网络末端 |
| 池化层 | $K \times K \times H_{out} \times W_{out}$（Max）| 无参数，少量运算 |
| ReLU | $C \times H \times W$ | 极少，几乎可忽略 |
| BatchNorm | $4 \times C \times H \times W$ | 较少，通常不占主导 |

---

### 11.9 Params 与 FLOPs 的关系与区别

| 对比 | Params（参数量） | FLOPs（计算量） |
|------|----------------|----------------|
| 衡量维度 | 存储空间 / 模型大小 | 推理速度 / 计算开销 |
| 单位 | M（Million） | GFLOPs / MFLOPs |
| 与 H×W 的关系 | **无关** | **有关**（空间越大，计算越多） |
| 典型瓶颈 | 内存/显存受限时关注 | 推理延迟受限时关注 |

**Params 少 ≠ FLOPs 少**：
- 深度可分离卷积（Depthwise Separable Conv，MobileNet）大幅减少 FLOPs，但 Params 减少比例不同
- 1×1 卷积：Params 不大（核小），但通道数大时 FLOPs 仍可观

> **考试思路**：给定网络结构，能算出一层的 Params 和 FLOPs；理解二者的区别。