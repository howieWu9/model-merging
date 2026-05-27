---
title: domain shift

---

# domain shift 

## Bridging Domains through Subspace-Aware Model Merging

CVPR 2026







## 文章思路

不同 domain 的模型虽然来自不同数据分布，但它们做的是同一个分类任务，所以它们学到的方向很容易相互重叠。

这种重叠会让 多个模型在相似 singular directions 上竞争，导致 merging 时发生 subspace conflict。

如何把多个在不同 domain 上 fine-tuned 的模型合并成一个模型，并让它在 unseen domain 上也表现更好。






## 文章的做法

**目的**： 在合并模型时，减少不同 domain 的 singular directions 冲突。

给定domains: $\{S^d\}_{d=1}^D$，保留一个domain $S^t$ 作为 unseen target domain, 剩下的 $D - 1$ 个 domains 用来合并模型。

$$\Delta w_d = \theta_d - \theta_{\text{pre}}$$

$$\mathrm{SAR}(\Delta w_i, \Delta w_j; k_j) = \frac{ \left\| \Pi_{k_j,j} \Delta w_i \right\|_F }{ \left\| \Delta w_i \right\|_F
}, \quad where \quad
\Pi_{k_j,j} = U_{k_j, j} U_{k_j, j}^\top 
\tag{1}$$


SAR函数用来衡量 $\text{不同 domain 的 } \Delta w \text{ 子空间重叠程度}$，目的是抛出文章的动机，它并不参与融合阶段的算法





$$k_j = \min \left\{ k : \left\| \Delta w_j - \Pi_{k_j,j}\Delta w_j \right\|_F \leq \epsilon \left\| \Delta w_j \right\|_F \right\} = \min \left\{ k : \frac{
\sum_{i=k+1}^{r}\sigma_i^2 }{ \sum_{i=1}^{r}\sigma_i^2 }
\leq \epsilon^2 \right\}\tag{2}$$


$$\Delta w_d^{(l)} \quad \text{简化层上标后：} \Delta w_d = U_d \Sigma_d V^{T}_d$$

$$U_{*} \leftarrow [ \,U_1 \mid \dots \mid U_D \,] \qquad
V_{*} \leftarrow [ \,V_1 \mid \dots \mid V_D \,]$$

$$U_* = P_{U_*} \Sigma_{U_*} Q_{U_*}^{T}, \qquad V_* = P_{V_*} \Sigma_{V_*} Q_{V_*}^{T}$$

$$U_\perp \leftarrow P_{U_*} Q_{U_*}^{\top},
\qquad V_\perp \leftarrow P_{V_*}Q_{V_*}^{\top} \tag{3}$$

$$\Delta'_d = U_\perp^\top \Delta_d V_\perp \qquad (\Delta_d = \Delta w_d)$$

$$\text{trim}(\Delta'_k)_{ij} =
\begin{cases}
(\Delta'_k)_{ii}, 
& \text{if } i = j 
\quad \text{Keep the diagonal}
\\
(\Delta'_k)_{ij}, 
& \text{if } i \neq j 
\text{ and }
\left|(\Delta'_k)_{ij} - \mu_{\text{off}}\right|
<
\tau \cdot \sigma_{\text{off}}
\\
0, 
& \text{otherwise}
\quad \text{Prune outliers}
\end{cases} \tag{4}$$

$$\Sigma_{\text{score}} = \sum_{d=1}^{D} \text{trim}(\Delta'_d) \to 
\hat{M} = U_\perp \Sigma_{\text{score}} V_\perp^\top \to
\theta_{\text{final}} = \theta_{\text{pre}} + \hat{M}$$





公式（1）：子空间对齐率(SAR)衡量 模型 $i$ 的参数更新 $\Delta w_i$ 投影到模型 $j$ 的主要子空间里，看有多少比例可以被 $\Delta w_j$ 的 top-$k_j$ singular subspace 表示。


公式（3）：因为 $U_*$ 和 $V_*$ 内部各向量之间不再保证正交。所以，再次对 U V进行SVD，得到一个正交化后的矩阵 $U_\perp, \quad V_\perp$。
它们构成一个共享的输入 ($U_\perp$)和 输出基($V_\perp$)，最接近所有 $D$ 个特定领域的子空间。





* $k_j$: 指从模型 $j$ 的奇异值分解（SVD）中保留的前 $k$ 个主奇异向量的数量。
* $\Pi_{k_j,j}$: 一个 projection matrix。由 $\Delta w_j$ 的前 $k_j$ 个左奇异向量所张成的子空间.
* $\|\cdot\|_F$: Frobenius 范数，用于计算矩阵所有元素平方和的平方根.
* $\epsilon$: 近似误差阈值（Hyperparameter），决定了允许丢失多少信息.
* $\sigma_i$: 矩阵 $\Delta w_j$ 的第 i 个奇异值.
* $d, D$: 分别表示 第 $d$ 个 domain 和 domain 总数。
* $\Delta_d$:  第 $d$ 个 domain/task 的**原始更新矩阵** (预训练W - 当前domain/task的模型W).
* $\Delta'_d, \,or \, \Delta'_k$: 把 **原始更新矩阵** 换到共享子空间坐标系后的矩阵
* $\mu_{\text{off}}$: 所有 $\Delta'_k$中 非对角元素的平均值（Mean）
* $\sigma_{\text{off}}$: 所有 $\Delta'_k$中 非对角元素的标准差（Standard Deviation）
* $\tau$: 文章设置为 1.96，因为它对应于标准正态分布的 95% 置信区间。
* $\hat{M}$: 最终 merged 的多domain 的update 矩阵。











