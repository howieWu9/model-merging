---
title: Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

---

# Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing
MENGYI YAN
August 2024 ACM SIGKDD （CCF-A & CORE A*）

Code, datasets and full version： https://github.com/authurlord/MELD


## key concepts
* 查询（Query）可以被视为一种特定格式的样本（Sample），文章将数据预处理查询（DP Query） q 定义为一个四元组 ：
    1. 自然语言指令（instruction）: 明确具体的任务（例如“实体匹配”）
    2. 演示示例（demonstrations）: 用于上下文学习（In-Context Learning）的一组带标签例子
    3. 条目/元组 (tuple/entry): 待处理的原始数据点
    4. 预期输出域（output domain）: 该任务期望的输出范围（如“匹配”或“不匹配”）


## 文章的方法

**专家精炼 (Expert Refinement) 优化目标**
> $$\arg \min_{\theta_{M_{RAG}}} \max_{\theta_{M_G}} I(M_G(X_i); M_G(RAG(X_i))) \qquad (1)$$
> 基于信息瓶颈（Information Bottleneck, IB）理论，通过计算互信息（Mutual Information）并采用 Min-Max 优化策略来寻找最优的参数组合


**路由网络 (Router Network) 优化目标**
>  $$\max \sum_{e_i \in \mathcal{N}(q_u)} I\!\left(e_i(q_u^i);\, l_u^i\right);
\quad
\min \sum_{\substack{e_i,e_j \in \mathcal{N}(q_u)}}^{i \neq j}
I\!\left(e_i(q_u^i);\, e_j(q_u^j)\right) \qquad (2)$$
> 第一部分 最大化 专家与标签之间的互信息（最relevant），第二部分 最小化 专家对 的互信息（保证diversity，即不同专家间相对无关联，不要“一堆回答都差不多”的专家）


(1): 通过一个基于信息瓶颈的 min–max 设计，同时优化 expert 模型 $M_\tau$ 和 RAG 增强过程，使 expert 能从增强数据中学到有用信息，同时避免 few-shot 偏置和重复增强带来的过拟合
对 $\theta_{M_{RAG}}$ 做最小化，对 $\theta_{M_\tau}$ 做最大化


(2): 是 MELD 中 router network 的核心训练目标：它通过 最大化被选专家对标签的互信息(relevance) 和 最小化被选专家之间的互信息(diversity) 这两个目标，迫使 router 为每个输入挑出一组既相关、又要彼此多样、低冗余（互补的）的 top-k experts

* $\operatorname{arg} min$: 找出 使目标函数取得最小值的 参数
* $M_G$: 专家所依赖的 base LLM / expert model。每个 expert 都是在一个基础 LLM 上进行参数高效微调得到的，$M_G$ 表示这个专家 backbone 的函数记号。
* $M_G(\cdot)$: 专家 backbone 模型的前向映射函数。它将输入数据映射到特征表示空间
* $M_{RAG}$: 指 MELD 系统中增强型检索增强生成（RAG）系统的骨干模型
* $\theta_{M_{RAG}}$: 微调后的 RAG 模型 ($M_{RAG}$) 的参数
* $\theta_{M_G}$: 表示 each expert 所基于的 基础大语言模型 (Base LLM) 的参数(tensor)，训练与 refinement 都是在这个模型上进行，用于构建专家
* $RAG(⋅)$：数据增强函数。它通过检索和元路径搜索，将原始输入转换为更丰富的增强样本
* $RAG(X_i)$: 表示通过 RAG 系统和元路径搜索增强训练数据
* $e_i, e_j$: 第 i 和 第 j 个专家。用来组成专家对 $(e_i,e_j)$，比较两者输出是否冗余
* $q_u$: 第 $u$ 个 query
* $q_u^i$: 原始样本 $q_u$ 在第 $i$ 个 expert / 第 $i$ 个任务视角下对应的 query 形式。
一个来自任务 $T_D$ 的 query 可以经过 self-annotation 或 task transformation，变换成适用于另一个任务/专家的 query-label pair；$q_u^i$ 就是这种 变换后的 query。
* $\mathcal{N}(\cdot)$: 稀疏门控路由函数。接收输入query，计算并输出选定的 Top-k 个专家的索引
* $l_u^i$: 对应于 $q_u^i$ 的标签（label）


