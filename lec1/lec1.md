# 第1讲：人工智能发展与学习范式演进

## 本讲提要

- 计算范式演进：计算 -> 逻辑计算 -> 机器计算
- 机器学习与深度学习的核心差异
- 人工智能“三起两落”历史脉络
- Hinton 对数字计算与生物计算的对比观点

---

## 1. 计算范式演进

从“计算”到“逻辑计算”再到“机器计算”，核心变化是：人类从手工设计算法，逐步走向让机器自动学习表示和规则。

里程碑事件：明斯基证明了单层神经网络不能解决异或（XOR）问题，这直接推动了后续多层网络与表示学习的发展。

## 2. 机器学习与深度学习对比

1. 传统机器学习常依赖人工特征工程：任务一变，特征通常要重做；性能瓶颈也常来自特征设计不足。
2. 深度学习强调表征学习：让模型从数据中自动学习多层特征，减少对人工特征的依赖。

## 3. 人工智能三起两落的历程

### 第一起：黄金时代（1956-1970）
- 1956年达特茅斯会议标志着人工智能作为独立学科的诞生
- 感知机的提出让人们对神经网络充满希望
- 逻辑推理、定理证明等早期AI研究取得进展
- 乐观主义盛行，学者们认为20年内机器将超越人类

### 第一落：第一次寒冬（1970-1980）
- 明斯基证明单层神经网络不能解决异或问题
- 早期AI系统能力有限，无法兑现承诺
- 政府和企业大幅削减AI研究经费
- AI进入低谷期

### 第二起：专家系统时代（1980-1987）
- 专家系统兴起，将人类专家知识编码为规则
- 神经网络研究复兴，Hopfield网络、BP算法相继提出
- 日本第五代计算机计划推动AI发展
- AI再次获得广泛关注和资金投入

### 第二落：第二次寒冬（1987-2006）
- 专家系统维护成本高、难以扩展
- 神经网络在复杂任务上表现不佳
- AI研究再次陷入低谷

### 第三起：深度学习时代（2006-至今）
- 2006年深度学习突破，Hinton提出深度信念网络
- 大数据时代提供海量训练数据
- GPU计算能力大幅提升
- ImageNet竞赛突破，AlphaGo战胜李世石等里程碑
- 大语言模型GPT系列引爆生成式AI热潮

## 4. Hinton 的观点（摘录）

- Digital computation requires a lot of energy but makes it very easy for agents that have the same model of the world to share what they have learned by sharing weights or gradients.
   Example: GPT-4 knows thousands of times more than any one person, using only about 2% as many weights.
- Biological computation requires much less energy but is much worse at sharing knowledge between agents.
   Final statement: If energy is cheap, digital computation is just better.

对应理解：数字计算的优势在于“知识可复制、可传播、可叠加”；生物计算的优势在于“低能耗”，但跨个体知识传递效率低。这一判断对大模型时代的工程路线具有很强解释力。