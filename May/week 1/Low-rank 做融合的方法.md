---
title: Low-rank 做融合的方法

---

# Low-rank 做融合的方法

## RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness

NeurIPS 2025

源代码：  https://github.com/AuroraZengfh/RobustMerge



## 方法

$$W = W_0 + \Delta W = W_0 + B \cdot A, \quad B \in \mathbb{R}^{d_o \times r} \quad A \in \mathbb{R}^{r \times d_i} \quad  r \ll \min(d_i,d_o) \tag{1}$$

$$W_m = W_0 + \lambda \sum_{n=1}^{N} \Phi(\Delta W_n)$$

$$\tilde{A} = \mathcal{M}_A(k)\odot A,\quad \tilde{B} = \mathcal{M}_B(k) \odot B \tag{2}$$

$$M_A(k)_{ij} =
\begin{cases}
0, & \text{if } |A_{ij}| \text{ is among the smallest } k\% \\
1, & \text{otherwise}
\end{cases}$$

$$S^i =
\frac{
\sum_{j=1}^{d_i} \mathrm{abs}(A[i,j])
}{
\sum_{j=1}^{d_i} \mathrm{abs}(\mathcal{M}_A[i,j]\odot A[i,j])
},
\quad i=1,\cdots,r
\tag{3}$$

$$\tilde{S}_n^i = \frac{S_n^i}{\sum_{n=1}^{N} S_n^i},
\quad n=1,\cdots,N \qquad \tilde{S}_n = \mathrm{Diag} (\tilde{S}_n^1,\tilde{S}_n^2,\dots,\tilde{S}_n^r) \tag{4}$$

$$\Delta \tilde W = \lambda \left(\sum_{n=1}^{N} (\tilde{B}_n \cdot \tilde{S}_n) \right) \cdot \left( \sum_{n=1}^{N} \tilde{A}_n \right) \tag{5}$$



公式（3）：**单个任务内**，通过 $A$ 的 pruning 前后幅度变化，估计每个 low-rank 维度（行）需要补偿（缩放）多少，然后把这个补偿应用到 $B$ 上。

公式（4）：**多个任务之间**，每个任务的补偿比例该如何平衡。


* $W_0$: 原始预训练模型里的权重矩阵，微调时被冻结。
* $\Delta W$: LoRA 学到的参数更新量
* $\Phi(\cdot)$ : 某种 merging 算法
* $\Delta W_n$:  第 $n$ 个任务的 LoRA 更新模块
* $k$: pruning rate
* $M_A(k)$: 作用在 A 上的 binary mask 矩阵
* $S^i$: diagonal matrix $S$ 对角线上的第 $i$ 个元素, 也就是第 $i$ 个 low-rank 维度的缩放系数

