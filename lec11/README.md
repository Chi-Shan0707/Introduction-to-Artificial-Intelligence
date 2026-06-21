# 第 11 讲：Transformer 与大模型基础

> 课程：Introduction to Artificial Intelligence · 复旦大学  
> 日期：2026-05-14  
> 资料：人工智能-11-transformer.pdf、本讲 Markdown/实践材料

## 本讲定位

系统学习自注意力、多头注意力、位置编码和 Transformer 架构，并连接 BERT/GPT/MOSS 等大模型。

## 学习目标

- 掌握 Q/K/V、自注意力和多头注意力的计算流程。
- 理解 Transformer 为什么比 RNN 更易并行、能捕捉长距离依赖。
- 区分 encoder-only、decoder-only 与 encoder-decoder 架构及其代表模型。

## 知识地图

| 模块 | 内容 |
| --- | --- |
| 自注意力 | 同一序列内部每个 token 与其他 token 计算相关性。 |
| 多头机制 | 不同 head 学习不同关系子空间，提高表达能力。 |
| 模型家族 | BERT 偏理解，GPT 偏生成，T5/原始 Transformer 支持 Seq2Seq。 |

## 核心概念

- Q 是查询，K 是可匹配的键，V 是被加权汇聚的信息。
- 位置编码弥补注意力本身不含顺序信息的问题。
- LLM 能力来自架构、数据规模、训练目标、对齐方法和推理策略共同作用。

## 课堂讲解补充

Transformer 的核心突破是用自注意力替代循环结构，使长距离依赖建模和并行训练同时成为可能。Q/K/V、多头注意力、位置编码、残差连接与 LayerNorm 共同组成大模型的基本模块。学习本讲要把公式和张量形状对应起来。

### 复习组织方式

- **问题背景**：这节课为什么要引入这个概念。
- **方法主线**：算法或模型按什么步骤工作。
- **公式/代码**：至少抓住一个能落地计算的表达。
- **局限性**：说明它在哪些场景下会失效或需要改进。

## 公式与记忆点

- Attention(Q,K,V)=softmax(QKᵀ/√d_k)V。
- 多头注意力中若 `d_model=12, h=3`，则每个头的维度 `d_k=d_v=d_model/h=4`。
- 位置编码加在输入 embedding 上：`X_pos = X + PE`，不是分别加到 `Q/K/V` 上。
- 残差结构：x_{l+1}=LayerNorm(x_l+Sublayer(x_l))。

## 典型试题：多头注意力的张量尺寸与参数量

设输入序列长度 `n=5`，batch size `B=2`，词向量维度 `d_model=12`，注意力头数 `h=3`。每个头的维度为：

```text
d_k = d_v = d_model / h = 4
```

### 1. Q、K、V 的尺寸

先对输入 `X∈R^{B×n×d_model}` 做线性投影：

```text
Q = XW_Q + b_Q
K = XW_K + b_K
V = XW_V + b_V
```

整体尺寸：

```text
X, Q, K, V: 2 × 5 × 12
```

拆成 `3` 个头后，每个头：

```text
Q_i, K_i, V_i: 2 × 5 × 4
```

### 2. 位置编码加在哪里

Self-Attention 本身对 token 顺序不敏感；若交换输入 token，同时交换对应行，注意力计算并不知道“谁在第几个位置”。因此要在进入 Q/K/V 投影之前加入位置编码：

```text
X_pos = X + PE
```

位置编码基础尺寸：

```text
PE: n × d_model = 5 × 12
```

实际与 batch 相加时广播为：

```text
2 × 5 × 12
```

标准 sinusoidal positional encoding 为：

```text
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

其中 `pos∈{0,1,2,3,4}`，`i` 是维度对的索引。若题目强调“cos 位置编码”，通常只取余弦分量：

```text
PE(pos, i) = cos(pos / 10000^{i/d_model})
```

但标准 Transformer 版本必须同时写出 sin/cos 交替形式。

### 3. 注意力权重与每个头输出

对第 `i` 个头：

```text
Score_i = Q_i K_i^T / sqrt(d_k)
```

尺寸计算：

```text
Q_i:     2 × 5 × 4
K_i^T:   2 × 4 × 5
Score_i: 2 × 5 × 5
```

注意力权重矩阵：

```text
A_i = softmax(Score_i, dim=-1)
A_i: 2 × 5 × 5
```

每一行表示“某个 query token 对全部 key token 的注意力分布”，因此最后一维 softmax 后每行和为 `1`。

每个头输出：

```text
Head_i = A_i V_i
Head_i: (2 × 5 × 5)(2 × 5 × 4) = 2 × 5 × 4
```

三个头拼接：

```text
Concat(head_1, head_2, head_3): 2 × 5 × 12
```

再经过输出投影 `W_O` 后仍为：

```text
2 × 5 × 12
```

### 4. 参数量

题目说明 `W_Q/W_K/W_V/W_O` 均保持维度不变且包含偏置。每个线性层：

```text
weight: 12 × 12
bias:   12
参数量: 12×12 + 12 = 156
```

所以：

| 线性层 | 参数量 |
|---|---:|
| `W_Q` | `156` |
| `W_K` | `156` |
| `W_V` | `156` |
| `W_O` | `156` |
| 仅 Q/K/V 三个投影 | `468` |
| 完整多头注意力投影层 | `624` |

注意：有些题目会把 `W_Q/W_K/W_V` 合并写成一个大矩阵 `W∈R^{12×36}`，再一次性生成 `QKV`。参数量仍然是：

```text
12×36 + 36 = 468
```

与三个独立线性层完全等价。

## 易错点

- 把 attention 理解成固定规则；它是由数据训练出的动态加权。
- 忽视 mask：生成式模型必须防止当前位置看到未来 token。

## 课后巩固

- 手算一个 2-token、1-head 的注意力权重。
- 比较 BERT 和 GPT 的训练目标与使用场景。

## 文件索引

| 文件 | 说明 |
|---|---|
| `lec11_notes.html` | 详细课程笔记网页 |
| `lec11.md` | 本讲补充材料 |

## 关键词

`Self-Attention`, `Multi-Head`, `Positional Encoding`, `Transformer`, `BERT`, `GPT`, `LLM`
