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
- 残差结构：x_{l+1}=LayerNorm(x_l+Sublayer(x_l))。

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
