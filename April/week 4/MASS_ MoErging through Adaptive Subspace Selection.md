---
title: 'MASS: MoErging through Adaptive Subspace Selection'

---

# MASS: MoErging through Adaptive Subspace Selection

ICLR 2026

源代码（目前还没删库跑路 23 Apr 2026）：https://github.com/crisostomi/mass





## key concepts

1. 头/head: 指的就是encoder，负责把输入数据变成embed 成 vector






## our problem

1. 并不是所有的模型（备选）都有 merging的意义
2. merge后参数变化 + -？
3. 是不是意味着我的模型可以缩小模型参数规模，但是又可以达到同样的性能/接近
4. 如果有domain shift怎么办？






## key assumptions

换problem setting？？？？？？？？？？

本文假设推理时任务未知 ，并且最合适的编码器子空间和相应的分类头都必须自动确定。这种设定使得单个通用模型能够在微调过程中处理所有遇到的任务，而无需外部监督。







## background

1. task vector 法的 model merging

$$\theta_{MT} = \theta_{pre} + \alpha \sum_{i=1}^T \tau_i$$

2. Task Singular Vectors 法: 对第 $i$ 个任务矩阵 $\Delta_i$ 做SVD，进行低秩近似 并正交化它们的奇异向量以减少任务间干扰

$$\theta_{MT}^{(l)} = \theta_{pre}^{(l)} + \alpha \sum_{i=1}^T \Delta_i^{(l)}, \quad 
where \quad \Delta_{i}^{(l)} = \theta_{pre}^{(l)} - \theta_{ft_i}^{(l)}$$

$$then, \text{generic layer task singular vector} \to \Delta_i = U_i \Sigma V_i^\top$$

* $\Delta_i^{(l)}$ 称为任务 $i$ 的逐层任务矩阵，当第 $l$ 层具有矩阵结构时







## approach

### 文章思路

总体思路是：
先做个一次性融合，得到多任务融合模型 $\theta_{MT}$。
然后，根据输入，判断对于这个输入，真正最适合模型的是哪一些 fine-tuned 模型。
最后，把这些适合当前输入的微调模型的task vector求出来，和预训练模型合并起来。

1. pre-processing step

    先做一次性的 merge，得到一个编码器模型 $\theta_{MT}$。

2. 推理阶段

    在测试时，再基于输入样本动态处理:
    
    * 第一遍前向传播：把输入送进 $\theta_{MT}$，在某一层取出 embedding $z_l$；
    * Routing：把 $z_l$ 投影到各个 task subspace 上，评估整个任务集合，看哪个投影误差最小；
    * Adaptive merge：把上一步被选中的 task subspace 合并成 $\Delta_{ada}$
    * 最后，只动态融合对当前输入最有用的那些任务对应的 subspace: $\theta_{MT} = \theta_{pre} + \alpha \Delta_{ada}$






### 3.1 fixed merging

$$\theta_{MT} = \theta_{pre} + \alpha \sum_{i=1}^{T} \mathbf{1}_{[g_i(X)=1]}(X) \tau_i = \theta_{pre} + \alpha \sum_{i=1}^{T} \mathbf{1}_{[g_i(X)=1]}(X) \sum_{j=1}^{k} \sigma_j^{i} u_j^{i} v_j^{i\top} \tag{1}$$

公式（1）：$\tau_i \approx \sum_{j=1}^{k} \sigma_j^i u_j^i v_j^{i\top}$ 说明第 $i$ 个任务更新可以用前 $k$ 个最重要的奇异方向来近似。这个思想就是 **Task Singular Vector** 这种方法的核心基础。

* $\mathbf{1}_{[g_i(x)=1]} (X)$: 指示函数, 表示第 $i$ 个任务是否被当前输入 $X$ 激活
* $g_i(X)$: 第 $i$ 个任务对应的 gating function
* $\tau_i$: 第 $i$ 个任务的 task vector
* $k$: 保留的 **奇异分量 数量**, 控制压缩程度
* $\sigma_j^i$: 第 $i$ 个任务、第 $j$ 个奇异值
* $u_j^i$: 第 $i$ 个任务、第 $j$ 个 左奇异向量
* $v_j^{i\top}$: 第 $i$ 个任务、第 $j$ 个 右奇异向量 的转置





### 3.2.1 Projection-based routing

把 $z_l$ 分别投影到每个任务($i = 1 \to k$)的右奇异向量子空间上，然后看 **投影后的重构误差(residual)** 有多小。误差越小，说明这个输入越适合那个任务子空间。

$$r_i = \|z_l - \operatorname{Proj}_{V_i^{(l)}} (z_l)\|_2, \quad where \quad 
\operatorname{Proj}_{V_i^{(l)}} (z_l)=V_i^{(l)}\big(V_i^{(l)}\big)^\top z_l \tag{2}$$

* $r_i$ : 第 $i$ 个任务的 residual
* $z_l$ : 输入样本在第 $l$ 层的中间表示(embedding)
* $V_i^{(l)}$ : 一个矩阵，表示 第 $i$ 个任务在第 $l$ 
* $Proj_{matrix}(vector)$: 是 $z_l$ 在 $span(V_i^{(l)})$ 上的正交投影。
* $span(\cdot)$: 表示由矩阵列向量所张成的所有线性组合的集合
* $span(V_i^{(l)})$ 是由 $V_i^{(l)}$ 的列向量张成的子空间。 表示第 $i$ 个任务的重要特征方向集合。





### 3.2.2 冗余指令的处理

有些任务彼此非常相似，它们的子空间会共享很多方向。这样一来，这些相似任务 合起来 会显得更强、更宽，更容易把输入解释掉，于是 router 会更偏向它们；反过来，一些较少见但确实重要的任务子空间就可能被压制。

所以作者在 fixed merging step 里, 不是所有 task update 都拿来 merge。如果一个新任务更新和已经接受的某个更新相似度高，就把它丢掉；只保留 足够不同 的那些 task matrices。

$$\|z_l - \operatorname{Proj}_{V_{MN} \cup  V_{EMN}}(z_l)\|_2 < \|z_l - \operatorname{Proj}_{V_{KM}}(z_\ell)\|_2 \quad where \quad \mathrm{span}(V_{MN}^{(l)})  \approx \mathrm{span}(V_{EMN}^{(l)}) \tag{3}$$

$$let \{\Delta_{a_1},\dots,\Delta_{a_r}\} \quad \delta_i=\operatorname{vec}(\Delta_i) \quad so, \quad\max_{1 \le m \le r} \operatorname{sim}(\delta_i,\delta_{a_m}) > \varepsilon, \tag{4}$$


* $\mathrm{span}(\cdot)$: 张成空间, 表示由矩阵列向量所张成的所有线性组合的集合
* $V_{MN}\cup V_{EMN}$: 这里不是严格集合论意义上的并集公式推导, 而是 把两个任务的方向一起考虑， 形成更大的联合子空间。
* $\{\Delta_{a_1},\dots,\Delta_{a_r}\}$: 一组矩阵, 表示 当前已经接受的 task updates 集合；用来 作为 已保留 的任务更新集合，用来和新来的 $\Delta_i$ 比较。
* $a_r$ 是 第 r 个已经被接受的任务索引。
* $\operatorname{sim}(\delta_i,\delta_{a_m})$: $\delta_i$ 与已接受更新 $\delta_{a_m}$ 的 cosine similarity
* $\varepsilon$: 设定的相似度阈值






### 3.3 Adaptive Merging and Inference

router 选择了一个相关任务子集 $\Omega$ 后，如何把这些任务的子空间动态合并成当前样本专属模型 $\theta_{MASS}$，并通过比较各候选 head 的 logits 来完成未知任务场景下的最终分类。

作者的方法：
所有被路由选中的候选任务分类头 $\{h_i\}_{i \in \Omega}$ 都会接收到完全相同的特征表示 $z_{L−1}$ 作为输入; 模型会为每一个选中的任务 $i$ 计算其分类结果:

$$let \quad i \in \Omega, \quad then \quad z_i = h_i(z_{L-1}), \qquad z_i \in \mathbb{R}^{C_i} \tag{5}$$

然后，我们从 $\Omega$ 的所有头部中选择 logit 值最高的头部，即：
 
$$(i^\star, c^\star) = arg \max_{(i,c)\in \Omega \times \{1,\dots,C_i\}} z_i[c] \tag{6}$$


* $\Omega$: router 选出的相关任务的索引集合
* $z_{L-1}$: 倒数第二层的共享表示(embedding of X)，即分类头之前的最后一层特征层;
* $h_i(\cdot)$: 第 $i$ 个任务的分类头; 输出 该任务上的类别 logits 向量
* $z_i$: 一个 $C_i$ 维向量; 指的是第 $i$ 个候选任务分类头 $h_i$ 的输出结果，即该任务在各个类别上的 Logits 向量
* $C_i$: 第 $i$ 个任务的类别数
* $(i^\star, c^\star)$: 最终选中的任务索引 $i^\star$ 和 类别索引 $c^\star$
* $\Omega \times \{1,\dots,C_i\}$: 所有被选中的任务 $i$，以及每个任务下的所有类别 $c$





### 3.4 残差最小化作为最大后验概率估计

这部分是对3.2路由部分的数学证明

