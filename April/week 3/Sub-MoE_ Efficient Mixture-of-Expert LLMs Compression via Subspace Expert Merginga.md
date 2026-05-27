---
title: 'Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merginga'

---

# Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merginga

Lujun Li1* , Qiyuan Zhu1* , Jiacheng Wang2, Xiaoyu Qin3, Wei Li4, Hao Gu1, Sirui Han1†, Yike Guo

AAAI 2026

没代码

## contribution
融合专家的同时，还缩小了模型规模(SVD)

## 文章思路
### 3.3

1.	先把多个 expert 的 $W$ 放在一起做联合分解
2.	得到共享的 $U,\Sigma$ 和各自的 $V_i$
3.	只在子空间里 merge $V_i$
4.	最后再重构出一个新的 $W_{merged}$

### 3.4

1. 先根据输入激活信息，为每个 expert 构造一个 activation-aware 矩阵 $S_i$ (用专家激活频率作为merging权重)
2. 用激活权重，通过加权求和的方式融合模型
3. 对融合后的模型参数做SVD
4. 在子空间里 merge 各个 $V'^{(i)}$
5. 对奇异值矩阵 $\Sigma'$ 做 truncation（截断），去掉较小的奇异值；然后重构矩阵 $W$






## 3.1 

$$y = \sum_{i=1}^{n} G_{i}(x) \cdot E_{i}(x), \quad E(x) = (\sigma(x \cdot W_{gate}) \odot (x \cdot W_{up})) \cdot W_{down} \tag{1}$$

* $G_i(x)$: router 给第 $i$ 个 expert 的路由权重(贡献比例)。
* $E_i(x)$: 第 $i$ 个 expert 对输入 x 的输出。
* $\sigma(\cdot)$: 激活函数。
* $W_{gate}$: expert 内部 gate 分支的权重矩阵
* $W_{\mathrm{up}}$: 专家内部 FFN 的升维投影权重矩阵
* $\odot$: 表示 Hadamard 积（对应元素相乘）（Hadamard product）。






## 3.2 

$$\mathrm{Sim}(E_i, E_j) = \frac{1}{m} \sum_{l=1}^{m} \frac{E_i(x_l) \cdot E_j(x_l)}{\|E_i(x_l)\| \cdot \|E_j(x_l)\|} \tag{2}$$

$$J=\sum_{i=1}^{k}\sum_{E_j\in Q_i}\|\mathcal{Y}_j-C_i\|^2, \quad where \quad \mathcal{Y}_i=\{E_i(x_1),\dots,E_i(x_m)\} \tag{3}$$


公式（2）：对于两个不同的专家，在同一批输入（m个）内，计算它们俩的 平均 cosine similarity，得到它们的功能相似度。


* $J$: 聚类的目标函数值，也可以理解为聚类误差、类内总平方距离。用一个单独的数来衡量当前聚类结果的好坏。
* $Q_i$:  表示分配给 第 $i$ 个cluster 的专家集合。
* $\mathcal{Y}_j$: 第 j 个专家对一批(size of m)输入样本的 输出集合
* $C_i$: 第 i 个 cluster 的中心（centroid）。







## 3.3

$$W_{merged} = \sum_{i=1}^{n} \alpha_i W^{(i)} \quad \tag{4}$$

### 用Sub-MoE法来改进 (4) 中的合并：

3.3 中公式（5）的 $W'^{(i)}$ 其实是排版错误，正确的应该是 $W^{(i)}$, 原因见3.4 第一行公式。这就是为什么公式（8）是 $U$ 而公式（11）是 $U'$

$$\mathrm{SVD} \left([W'^{(1)};\ldots;W'^{(n)}]\right) = U' \Sigma' [V'^{(1)};\ldots;V'^{(n)}]^{T} \tag{5}$$

$$f(V_i) = \frac{\sum_{x \in \mathcal{X}} \mathbb{I}[i \in TopK(G(x), k)]}{|\mathcal{X}|} \tag{6}$$

$$V_{merged} = \frac{\sum_{i \in Q} f(V_i) \cdot V_i}{\sum_{i \in Q} f(V_i)} \tag{7}$$

$$W_{merged} = U\Sigma[V_{merged}]^T \qquad \text{Sub-MoE 合并法} \tag{8}$$

公式（5）已经先做了联合 SVD, 得到共享子空间和每个 expert 对应的 V 块。 

公式（6）: 第 $i$ 个 expert 的激活频率（sampling frequency / activation frequency）

公式（7）: 利用公式（5）和（6），把同一个 cluster Q (文章的**Figure 2**)里的多个 expert 的 V （matrix），按照它们的激活频率做加权平均，得到一个合并后的 V（matrix）。

公式（8）: 融合其余专家，最终得到了一个新的 merged weight matrix



* $V$: 每个 expert 对应的 V 块
* $V'^{(i)}$: 对激活加权后的专家权重进行 SVD 分解得到的第 $i$ 个右奇异向量矩阵块。
* $W = [W'^{(1)};\ldots;W'^{(n)}]$: 把第 1 到第 $n$ 个 expert 的权重矩阵拼接起来形成的大矩阵。
* $\mathbb{I}[i\in TopK(G(x),k)]$: 判断 对于某个输入 $x$，expert $i$ 有没有进入 top-k。
* $Q$: 当前 cluster 中的 expert 集合。







## 3.4  Sub-MoE with Intra-Expert Compression

$$\text{re-weight each expert’s weight matrix} \quad \to \quad W'_i = W_i S_i$$

$$\mathrm{SVD} \left([W'^{(1)};\ldots;W'^{(n)}]\right) = U' \Sigma' [V'^{(1)};\ldots;V'^{(n)}]^{T} \tag{5}$$

$$V_{\mathrm{merged}} = \frac{\sum_{i\in Q} f(V_i)\cdot  V'^{(i)} S_i^{-1}} {\sum_{i\in Q} f(V_i)} \tag{10}$$

$$W_{merged}^{trunc} = U' \cdot Trunc.(\Sigma') \cdot V_{merged} \tag{11}$$


公式（10）：因为前面先把 $W_i$ 变成了 $W_iS_i$，所以后面 merge V 的时候，要乘矩阵S的逆 $S_i^{-1}$ 把 $W_i$ 换回来。


* $S_i$ : 激活加权矩阵 (activation weighted matrix), 对于专家权重矩阵 $W_i$，我们首先通过测量输入激活 $X_i$ 的相关性来获得激活加权矩阵 $S_i$。$S_i$ 有效地保留了突出重量并减少了分解误差。
* $V'^{(i)}$ : 对激活加权后的专家权重进行 SVD 分解得到的第 $i$ 个右奇异向量矩阵块。
* $U'$ : 通过对**激活加权后的**专家并集进行 SVD 分解得到。
* $Trunc(\cdot)$ : 该函数用于控制压缩率。它会舍弃（截断）$Σ'$矩阵中最小的那些奇异值。由于奇异值代表了信息的能量，舍弃较小的值可以在保留绝大部分模型能力的同时，显著降低模型所需的秩（Rank）(**这也说明模型参数的规模被缩小了**)












