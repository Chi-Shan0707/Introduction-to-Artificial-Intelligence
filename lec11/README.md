# Lecture 11 笔记：RNN、注意力机制与 Transformer

本讲围绕“序列建模”展开，从 RNN 的循环结构出发，逐步引出梯度问题、GRU/LSTM 的门控机制、Seq2Seq 与注意力，再过渡到 Transformer 的自注意力框架，并补充了多头注意力、位置编码、LayerNorm 以及线性层与 $1\times1$ 卷积的等价视角。

## 1. 序列建模任务类型
- One-to-One：单输入到单输出（如分类）。
- One-to-Many：单输入到多输出（如图像描述）。
- Many-to-One：多输入到单输出（如情感分类）。
- Many-to-Many：多输入到多输出（如翻译），可对齐或不对齐。

## 2. RNN 基础
### 2.1 递归与参数共享
RNN 的核心是“结构不变 + 参数共享”。每个时间步使用同一套参数，将当前输入与上一步隐藏状态融合：

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1})$$

输出常见两种方式：
- 直接输出：$y_t = h_t$。
- 线性映射：$y_t = W_{hy} h_t$。

### 2.2 展开视角
把 RNN 沿时间展开，相当于“无限层但参数相同”的深层网络，便于理解 BPTT（时间反传）。

### 2.3 激活函数
$\tanh$ 输出范围为 $(-1, 1)$，零中心，通常比 Sigmoid 更利于梯度传播。

## 3. 梯度消失/爆炸与长依赖问题
RNN 在长序列上会出现梯度指数衰减或爆炸，导致“记不住很久以前的信息”。这是引出门控结构（GRU/LSTM）与注意力机制的直接动机。

## 4. GRU：门控式记忆
GRU 引入两个门：更新门与重置门，决定“保留多少旧信息、吸收多少新信息”。

$$z_t = \sigma(W_z [x_t, h_{t-1}])$$
$$r_t = \sigma(W_r [x_t, h_{t-1}])$$
$$\tilde{h}_t = \tanh(W [x_t, r_t \odot h_{t-1}])$$
$$h_t = (1 - z_t) \odot \tilde{h}_t + z_t \odot h_{t-1}$$

GRU 结构更简洁、参数更少，训练速度快，性能常与 LSTM 相当。

## 5. LSTM：显式记忆通道
LSTM 增设了独立的细胞状态 $C_t$，并通过遗忘门、输入门、输出门控制信息流：

$$f_t = \sigma(W_f [x_t, h_{t-1}] + b_f)$$
$$i_t = \sigma(W_i [x_t, h_{t-1}] + b_i)$$
$$\tilde{C}_t = \tanh(W_C [x_t, h_{t-1}] + b_C)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$o_t = \sigma(W_o [x_t, h_{t-1}] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

门控结构提供“梯度高速通道”，显著缓解长依赖问题。

## 6. Seq2Seq 与 Teacher Forcing
### 6.1 Seq2Seq
Encoder 将输入序列压缩为上下文向量 $C$，Decoder 逐步生成输出：

$$s_t = g(y_{t-1}, s_{t-1}, C)$$

### 6.2 Teacher Forcing
训练时把“真实上一词”喂给 Decoder，而不是模型预测值，收敛更快，但可能带来 exposure bias。

### 6.3 上下文瓶颈
单一向量 $C$ 难以承载长序列全部信息，引出注意力机制。

### 6.4 过渡动机：从“循环”到“可查询的记忆”
即便引入 GRU/LSTM，RNN 仍有两个根本限制：
- **信息路径**：所有历史信息都必须挤进隐藏状态并逐步传递，长距离依赖难以保真。
- **计算路径**：序列只能一步步算，训练与推理难以并行。

注意力机制的直觉是把编码端的每个时间步都当成“记忆条目”，解码时不再只依赖最后的 $C$，而是**按需查询**全体记忆。这样一来，长序列的信息不会被提前压缩，模型在每一步都能聚焦真正相关的片段。

## 7. 注意力机制的演进
### 7.1 加性注意力（Bahdanau）
以视觉注意力为例：二维特征图 $z_{i,j}$ 与解码器状态 $h$ 做对齐：

$$e_{i,j} = v^T \tanh(W_h h + W_z z_{i,j})$$
$$a_{i,j} = \frac{\exp(e_{i,j})}{\sum_{x,y} \exp(e_{x,y})}$$
$$c = \sum_{i,j} a_{i,j} z_{i,j}$$

### 7.2 展平与置换不变性
二维特征展平成集合后，注意力本质是“加权求和的集合运算”，对顺序不敏感，因而需要位置编码。

### 7.3 点积注意力与缩放
点积注意力更快但易导致 Softmax 饱和，Transformer 引入缩放因子：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 7.4 过渡：注意力从“辅助模块”走向“核心计算”
当注意力可以高效地在全局范围内做匹配时，**循环结构就不再是必须**。RNN 的隐藏状态本质上充当了“查询向量”，如果查询能直接由序列本身生成，那么我们就可以让**每个位置都去查询全序列**，形成自注意力。再配合矩阵化计算与缩放点积，模型不仅能表达长依赖，还能获得更高的并行效率，这正是 Transformer 的核心出发点。

## 8. QKV 与自注意力
注意力最初是 RNN 的“外挂”，但当查询、键、值都能从同一序列中直接生成时，它就自然升级为**自注意力架构**，并逐步取代循环结构。

### 8.1 角色拆分
输入 $X$ 分别映射为：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

避免“同一个向量既做匹配又做内容”的限制。

### 8.2 Self-Attention 与 Cross-Attention
- 自注意力：$Q,K,V$ 全来自同一序列。
- 交叉注意力：$Q$ 来自 Decoder，$K,V$ 来自 Encoder。

### 8.3 并行优势
注意力可用矩阵乘法一次性计算所有位置的依赖，训练并行度高。

## 9. Multi-Head Attention
多个注意力头在不同子空间并行学习：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$$

类比 Inception 的多分支结构：不同头关注不同关系（语法、语义、位置等）。

## 10. Transformer 结构
### 10.1 Encoder
每层包含：
- Multi-Head Self-Attention
- 前馈网络（FFN）
- 残差连接 + LayerNorm

### 10.2 Decoder
每层包含：
- Masked Self-Attention（防止偷看未来）
- Cross-Attention
- FFN
- 残差连接 + LayerNorm

### 10.3 位置编码
自注意力对顺序不敏感，需加入位置编码（如正弦余弦编码）以注入顺序信息。

## 11. 注意力类型与模型范式补充
### 11.1 Self-Attention 与 Cross-Attention
- Self-Attention：$Q,K,V$ 来自同一序列，刻画序列内部依赖（Encoder 的核心）。
- Cross-Attention：$Q$ 来自解码端，$K,V$ 来自编码端，用于对齐源序列与目标序列（Seq2Seq 解码关键一步）。

### 11.2 Masked Self-Attention（因果掩码）
用于自回归生成场景（Decoder 或 Decoder-only）。通过掩码禁止关注未来位置，保证生成时序的因果性。

### 11.3 Encoder-Decoder（经典 Seq2Seq Transformer）
- Encoder：双向 Self-Attention，抽取全局语义表示。
- Decoder：Masked Self-Attention + Cross-Attention，逐步生成输出。
- 典型任务：机器翻译、摘要、对话生成、图像描述等“输入到输出”任务。

### 11.4 Encoder-only（双向编码）
- 只有 Encoder 堆栈，关注全局理解与表征学习。
- 典型任务：分类、序列标注、检索、语义匹配、向量表示（如 BERT 类）。

### 11.5 Decoder-only（自回归生成）
- 只有 Decoder 堆栈，使用因果掩码。
- 典型任务：语言模型、开放式生成、代码生成（如 GPT 类）。

## 12. 线性层与 $1\times1$ 卷积等价
从张量维度看，Transformer 的 `nn.Linear` 与 CNN 的 $1\times1$ 卷积都是“逐位置的通道变换”。

## 13. LayerNorm vs BatchNorm
统一公式：

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

- BatchNorm：跨 batch 统计，受 batch size 影响大。
- LayerNorm：对单样本特征维度统计，适合可变长度序列。

## 14. 课程小结
- RNN 通过循环结构处理序列，但存在梯度问题。
- GRU/LSTM 通过门控机制改善记忆与梯度传播。
- 注意力机制缓解 Seq2Seq 的信息瓶颈。
- Transformer 以自注意力为核心，完全摆脱循环结构，实现高并行训练。

## 15. 课后作业
自行探索一个关于 Transformer 的交互式学习工具，加深对注意力与模块结构的理解。