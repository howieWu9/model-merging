---
title: multiple view 方法

---

# multiple view 方法

## Deep Multiview Clustering by Contrasting Cluster Assignments

ICCV 2023

源代码： https://github.com/chenjie20/CVCL


## 问题：如何把这个方法用在model merging？也就是，在融合前，怎么挑选需要融合的专家/model 上


专家/models = views

$logits =  X^{(v)}$ 

$softmax(X^{v}) \to H^{(v)}$

或者

专家/models = views

$\Delta W^{(v)} \to SVD \to X^{(v)} = \text{span}(U_r^{(v)})$

算子空间重叠度/相似程度，然后找 $sim_{ij} = s(X^{(v_i)}, X^{(v_j)})$

根据重叠程度，做clustering

设计一个选择策略，在每个cluster选 expert/model

然后跳到 merging 阶段







## method

$$\mathcal{X} = \{ X^{(v)} \in \mathbb{R}^{d_v \times N} \}_{v=1}^{n_v} \qquad
X^{(v)} = \{ \mathbf{x}_1^{(v)}, \cdots,  \mathbf{x}_N^{(v)}\} \qquad
f: \mathcal{X} \to \{ Z^{(v)} \in \mathbb{R}^{N \times k} \}^{n_v}_{v=1}$$

$$f_{\{W_h^{(v)}\}_{v=1}^{n_v}}:\{Z^{(v)}\}_{v=1}^{n_v}\rightarrow \{H^{(v)}\}_{v=1}^{n_v}, \qquad where \quad H^{(v)}\in \mathbb{R}^{N\times K} \tag{1}$$

$$p_{ij}^{(v)} = \frac{ \left(h_{ij}^{(v)}\right)^2 / \sum_{i=1}^{N} h_{ij}^{(v)} }{ \sum_{k=1}^{K}
\left( \left(h_{ik}^{(v)}\right)^2 / \sum_{i=1}^{N} h_{ik}^{(v)} \right) } \tag{2}$$

$$s\left(p_j^{(v_1)},p_j^{(v_2)}\right) = \left( p_j^{(v_1)} \right)^T p_j^{(v_2)} \tag{3}$$

$$ l^{(v_1,v_2)} = -\frac{1}{K} \sum_{k=1}^{K} \log
\frac{ e^{s\left(p_k^{(v_1)},p_k^{(v_2)}\right)/\tau} }{T} \tag{4}$$

$$T = \sum_{j=1,j\neq k}^{K}
e^{s\left(p_j^{(v_1)},p_k^{(v_1)}\right)/\tau}
+
\sum_{j=1}^{K}
e^{s\left(p_j^{(v_1)},p_k^{(v_2)}\right)/\tau}$$

$$L_c = \frac{1}{2}
\sum_{v_1=1}^{n_v}
\sum_{\substack{v_2=1\\v_2\neq v_1}}^{n_v}
l^{(v_1,v_2)} \tag{5}$$

$$L_a = \sum_{v=1}^{n_v} \sum_{j=1}^{K} q_j^{(v)}\log q_j^{(v)}, \quad where \quad
q_j^{(v)} = \frac{ \sum_{i=1}^{N}p_{ij}^{(v)} }{N} \tag{6}$$


$$L_{fine}=L_{pre}+\alpha L_c+\beta L_a
\tag{13}$$

$$y_i = \arg \max_j \left( \frac{1}{n_v} \sum_{v=1}^{n_v} q_{ij}^{(v)} \right) \tag{14}$$



公式（1）：对每一个 view 的**语义特征矩阵** $Z^{(v)}$，通过一个带参数 $W_h^{(v)}$ 的映射函数 $f$ （编码器），输出该 view 的聚类概率矩阵 $H^{(v)}$


公式（2）：用 $H^{(v)}$ 里的概率 $h_{ij}^{(v)}$，重新计算一个 target distribution $P^{(v)}$ 里的元素 $p_{ij}^{(v)}$。


公式（3）：在不同view（$v_1, v_2$）中，计算同一个簇 $j$ 的 cluster 相似度。


公式（4）：是一个 cross-view contrastive loss。目标是：让两个 view 中同一个 cluster 的 assignment 更接近，同时让不同 cluster 的 assignment 更远。


公式（5）：对所有不同 view 之间的 contrastive loss $l^{(v_1,v_2)}$ 求和，得到总的跨视图对比损失


公式（6）：$L_a$ 是正则化项。如果很多样本都被分到第 $j$ 个 cluster，那么 $q_j$ 就会越大；这导致最后的总损失 $L_a$ 也越大。
公式（14）：对第 $i$ 个样本，把它在所有 view 中属于 cluster $j$ 的概率拿出来，然后取平均；最后选择平均概率最大的 cluster $j$，作为最终预测标签 $y_i$。



* $k$: 簇数量
* $Z^{(v)}$: 表示第 $v$ 个 view 的 semantic features，也就是 encoder 提取出来的特征矩阵。
* $W_h^{(v)}$: 表示第 $v$ 个 view 的 clustering head 的可学习参数。
* $\tau$: 为温度参数
* $q_j^{(v)}$: 表示第 $v$ 个 view 中，第 $j$ 个 cluster 的平均 assignment probability。


