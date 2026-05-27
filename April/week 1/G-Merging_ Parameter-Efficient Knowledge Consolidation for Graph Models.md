---
title: 'G-Merging: Parameter-Efficient Knowledge Consolidation for Graph Models'

---

# G-Merging: Parameter-Efficient Knowledge Consolidation for Graph Models

Jun Chen, Ziyue Qiao, Qin Zhang, Kaize Ding Xiao Luo

2026, accepted by ICLR

源代码：https://github.com/cjcj46262/G-Merging


## 关注点
1. 这篇文章在做merging时，是：base model + fine-tuned model = 一个多任务model
2. 模型参数的规模 不变
3. domain shift问题：
* 表示对齐（消除偏差）：利用 TWD（拓扑感知 Wasserstein 距离） 损失函数，在不使用标签的情况下训练适配器，强制让融合模型在处理新领域数据时的特征分布向原始专家的分布靠拢，从而校正表示偏差

* 动态路由: 在推理时，无需训练的路由器会根据输入图的拓扑结构，计算其与各个专家适配器之间的相似度。如果新数据表现出特定的领域特征，路由器会动态地给与其最相似的专家分配更高的权重，从而可以利用相似任务的经验来增强对新偏移数据的泛化能力

* 拓扑感知约束：TWD 损失强制要求特征传输必须遵循图的物理连接，这使得模型能够捕捉并保留特定领域的结构模式，从而比普通模型合并方法更能抵抗领域偏移带来的性能衰退




## 文章思路
这篇文章研究的是 graph model merging（图模型合并） 问题，而且是 pretrain-finetune 范式下的多任务知识整合。作者的出发点是：在图学习中，通常先有一个预训练 GNN，然后针对不同下游任务分别做 fine-tuning，最终得到很多个 task-specific 模型；但如果部署时每个任务都保留一个完整模型，会带来很高的存储成本和部署成本。因此，他们希望把这些 fine-tuned GNN 模型合并成一个统一模型。



## key concepts

1. 运输计划: 一个数学方案，描述了如何将一个概率分布（源分布）中的“质量”重新分配到另一个分布（目标分布）中
2. 质量：指概率权重或数据点的分布量。对于包含 $∣V∣$ 个节点的图，通常假设每个节点的初始**质量是均等的**，即每个点的**权重为 $\frac{1}{∣V∣}$**
3. 耦合（coupling）: 它定义了两个分布之间每一个点对点的质量流向，是计算分布之间相似性的核心数学工具
4. GNN: graph neural network
5. domain shift(distribution shift): 指模型在训练时接触的数据分布，与实际应用（测试）时遇到的**数据分布不一致**的现象
6. unified model：统一模型通过合并多个模型得到了各行各业的常识，但在处理特定任务时，它的表现不够精准，会有一些偏差
7. Correction Term（修正项）：为了让 unified model 的回答变成专家（fine-tune model）的回答，我需要对原始回答做多少修正
8. task vector：
9. 修正量：合并了任务向量后，统一模型生成的特征分布与原始微调模型（专家模型）之间存在的差异
10.  L1 距离：两个向量 $P(x_1, y_1)$ 和 $Q(x_2, y_2)$，L1 距离的计算公式是：$$∥(P, Q)∥_1 = |x_1 - x_2| + |y_1 - y_2|$$



## 文章问题
**figure 1 - （2）** 对 8 个分子图数据集的嵌入进行降维，结果显示不同领域的图在空间中呈现出截然不同的聚类模式，证明了图数据的结构异质性
图数据具有很强的结构异质性。当你简单地把多个模型合并在一起时(unified model)，不同任务之间的特征分布会产生冲突，导致合并后的模型生成的特征（表示）偏离了原始专家模型的轨迹
Adapter 的目标：Adapter $f_{ adap}$被训练用来捕捉这种偏差值（即统一模型输出与微调模型输出之间的差值）

**figure 1 - （3）** 跨域验证矩阵：展示了在一个数据集上微调的模型很难泛化到其他数据集，说明模型具有高度的领域特定性

**figure 1 - （3）** 性能对比：初步展示了 G-Merging 在 ROC 分数上明显优于简单的 权重平均（Weight Average）方法




## 文章方法
**figure 2**
* Phase I（参数合并）：利用任务算术（Task Arithmetic）将各任务向量合并，得到初步的统一模型，提取共享知识
* Phase II（引入适配器）：在冻结的统一模型上训练轻量级的 NodeAdapter 和 GraphAdapter。使用 TWD（拓扑感知 Wasserstein 距离） 进行节点级对齐，以及使用 L1 距离 进行图级对齐
* Phase III（推理阶段）：将适配器集成到 MoE（混合专家）架构中。输入图通过一个无需训练的路由器，根据拓扑相似性动态分配各个适配器的权重。






### Wasserstein 距离定义
$$D_{wd}(\mu, \nu)
= \min_{T \in \Pi(u,v)}
\sum_{i=1}^{n} \sum_{j=1}^{m} T_{ij} \cdot c(x_i, y_j) \qquad (1)$$
> 路径$(x_i,y_j)$的总成本 =  搬运的质量T $\times$ 单位运输代价c(x,y)

$$\sum_{i=1}^{n} u_i \delta_{x_i} = \mu \in P(X), \quad \sum_{j = 1}^{m} v_j \delta_{yj} = \nu \in P(Y)$$
 
$$\Pi(u,v)=\{T\in \mathbb{R}^{n\times m}_{+}\mid T1_m=u ∧ T^\top 1_n=v \}$$
公式（1）定义了两个离散概率分布 $\mu$ 和 $\nu$ 之间的 Wasserstein 距离：
作用是量化两个对象（如离散概率分布）之间的相似性。它通过计算从一个分布转移到另一个分布的最小最优传输成本来实现这一点。这个最小成本就是 Wasserstein 距离。

* $\mu = \sum_{i=1}^{n} u_i \delta_{x_i}$: 在空间 $X$ 上的离散概率度量（discrete probability measure），分别代表待比较的分布
* $\nu = \sum_{j=1}^{m} v_j \delta_{y_j}$: 
* $X, Y$: 两个分布各自定义的空间（sample space）
* $x_i$: 分布 $\mu$ 的第 $i$ 个支撑点（support point），在图模型背景下通常是节点嵌入向量
* $y_i$: 
* $1_n$:  $n$ 维全 1 向量
* $u, v$: 两个边缘分布权重向量
* $u_i$: 分布 $\mu$ 在支撑点 $x_i$ 处的权重
* $v_i$: 
* $\delta_{xi}$: 以 $x_i$ 为中心的 Dirac measure（狄拉克测度）
* $\delta_{yj}$:
* $\delta_{(xi, yj)}$: 对应点 $(x_i,y_j)$ 在联合空间上的 Dirac measure
* $\tau(x,y) = \sum_{i=1}^{n} \sum_{j=1}^{m} T_{ij}\,\delta_{(x_i,y_j)}$: $\mu$ 与 $\nu$ 的一个 coupling（耦合 / 联合概率分布）。它将两个独立的离散概率分布联系在一起，形成一个联合概率分布。它记录了为了让两个分布对齐，每一个数据点应该如何“移动”到对应的位置
    通过对带权重 ($T_{ij}$) 的狄拉克度量进行求和，就构建出了一个完整的离散联合概率分布
* $T_{ij}$: 从源点 $x_i$ 搬运到目标点 $y_j$ 的质量。$T$ 中记录了所有可能的 $n×m$ 种配对路径的运输方案
* $c(x_i, y_j)$: 把单位质量从 $x_i$ 搬到 $y_j$ 的 ground cost（基础运输代价）
* $Π(u,v)$: 所有满足联合分布约束的运输计划集合, 每个元素都是一个 矩阵 T (即, $T_{ij}$, 第 i 个源点到第 j 个目标点搬运多少质量)，用来限定哪些 $T$ 是合法的
* $D_{wd}(\mu, \nu)$: $μ$ 和 $ν$ 之间的Wasserstein 距离。目的是，在所有合法运输计划 $T$ 中找到总成本最小的那个。
* $T1_m = u$: 保证矩阵 $T$ 每一行的和等于 $u_i$，即：从源点 $x_i$ 送出去的总质量，必须恰好等于该点原本拥有的质量 $u_i$
* $T^\top1_n = v$: 保证矩阵 $T$ 每一行的和等于 $v_j$，即：送到目标点 $y_j$ 的总质量，必须恰好等于目标点需要接收的质量 $v_j$





### 4.2 incorporating  task-specific adapters
$$G = \{A, X\}, \qquad H^{(l)} \in \mathbb{R}^{|V| \times d}, \qquad H^{(0)} = X.$$

$$H_{\theta,\theta^\ast}^{(l)} = f_{conv,\theta}^{(l)} \!\left(A, H_{\theta,\theta^\ast}^{(l-1)}\right) -f_{adap,\theta^\ast} \!\left(f_{conv,\theta}^{(l)}
\!\left(A, H_{\theta,\theta^\ast}^{(l-1)}\right)\right),\quad H_{\theta}^{(l)} = f_{conv,\theta}\!\left(A, H_{\theta}^{(l-1)}\right) \qquad (2)$$

$$W_{down} \in \mathbb{R}^{d \times r}, \qquad W_{up} \in \mathbb{R}^{r \times d}, \qquad \theta^\ast=\{W_{down},W_{up}\}$$

$$f_{adap, \theta^*}(H) = \text{ReLU}(H \cdot W_{down}) \cdot W_{up} \qquad (3)$$

$$ h \in \mathbb{R}^{d}$$

$$h_{\theta, \theta^*} = f_r(\{H^{(l)}_\theta\}) - f_{adap, \theta^*}(f_r(\{H^{(l)}_\theta\})), \qquad h_\theta = f_r(\{H^{(l)}_\theta\}) \qquad (4)$$

$$\Pi(A) = \left\{T \in \mathbb{R}_{+}^{|\nu| \times |\nu|}  \middle|  T\mathbf{1}_{|\nu|} = \frac{1}{|\nu|} \cdot \mathbf{1}_{|\nu|} \wedge T^\top \mathbf{1}_{|\nu|} = \frac{1}{|\nu|}\mathbf{1}_{|\nu|} \wedge T \odot (\mathbf{1}_{|\nu|\times|\nu|} - A) = \mathbf{0}_{|V|\times|V|} \right\}$$

$$h_i^{(l)} = H_{\theta_{uni}, \theta_k^\ast}^{(l)}[i,:], \qquad h_j^{\prime(l)} = H_{\theta_k}^{(l)}[j,:]$$

$$\theta_{uni} = \theta_{pre} + \lambda \sum_{k=1}^{K} \tau_k, \qquad \tau_k = \theta_k - \theta_{pre}$$

$$\mathcal{L}_{TWD} = TWD(H^{(l)}_{\theta_{uni}, \theta^*_k}, H^{(l)}_{\theta_k}, A) = \min_{T \in \Pi(A)} \sum_{i=1}^{|\nu|} \sum_{j=1}^{|\nu|} T_{ij} \cdot c(h^{(l)}_i, h'^{(l)}_j), \quad where \quad c(a,b) =\frac{1}{2}(1−cos(a,b)) \qquad (5)$$

$$\mathcal{L} = \min_{\theta^*_k} \frac{1}{|\mathcal{D}_k|} \sum_{G \in \mathcal{D}_k} \left( \alpha \cdot \mathcal{L}_{MD} + \sum_{l=1}^L \mathcal{L}_{TWD} \right) \qquad (6)$$


**公式（2）** 的作用是：定义第 $l$ 层 GNN 在 带 NodeAdapter 和 不带 NodeAdapter 两种情况下，节点嵌入矩阵如何更新
**第一部分**
$f_{conv}$ 先得到该层基础 GNN 表示；$f_{adap}$  在这个基础表示上生成适配修正量。然后构造带 adapter 的节点表示，即：基础 GNN 表示 - adapter 修正项

**第二部分**
标准 GNN 更新规则，表示没有 adapter 时，当前层表示只由图卷积产生

**公式（3）** 的作用是：给出 adapter 模块 $f_{\mathrm{adap},\theta^\ast}$ 的一个具体实现方式。先把输入表示 $H$ 从原始维度 $d$ 降维到较小维度 $r$;再经过一个非线性激活函数 $ReLU$，增强表达能力;最后再从低维 $r$ 升维回原始维度 $d$。

**公式（4）** 定义 图级表示 在带或不带 GraphAdapter 时分别如何计算
带 GraphAdapter 时，先得到原始图级表示，再通过 adapter 生成一个图级修正项，并把这个修正项从原始图级表示中减去，从而得到校正后的图表示。
不带 GraphAdapter 时，直接对所有层/最终层的节点嵌入做 readout（例如平均池化、汇聚），得到一个图级向量

**公式（5）** 在图结构 $A$ 的限制下，计算嵌入分布 $H_{uni}$ 与专家嵌入分布 $H_k$ 之间的距离。
    作用是作为特征对齐损失（Feature Alignment Loss），在训练适配器时消除表示偏差。它通过计算统一模型与原始微调模型节点嵌入分布之间的最小传输成本，强制让合并后的模型在保留图拓扑结构特征的同时，向原始专家的表示靠拢。
$\Pi(A)$中的三个条件：
> 行和约束：确保每个源节点的质量被完全运出（第一个约束条件要求每个节点 $i$ 运输的总量必须正好等于它拥有的全部质量）

> 列和约束:  确保每个目标节点接收到正确的质量（确保了每个目的地 $j$ 收到的总量正好符合目标分布的要求）

> 拓扑约束： 如果图中两个节点不直接相连，它们之间就禁止任何质量传输

**公式（6）** 任务特定适配器的最终优化目标（损失函数）


* $\theta$: GNN 模型的参数
* $\theta^* = \{W_{down}, W_{up}\}$: GraphAdapter 的参数集合
* $G$: 输入图（input graph data），由 **邻接矩阵 $A$** 和 **节点特征矩阵 $X$** 组成。
* $H$: 表示节点嵌入矩阵
* $H^{(l)}$: 第 $l$ 层的节点嵌入矩阵（node embedding matrix）
* $H_{\theta, \theta^*}^{(l)}$: 带 adapter 时，第 $l$ 层输出的节点嵌入矩阵。也就是同时受 GNN 参数 $\theta$ 和 adapter 参数 $\theta^\ast$ 影响的表示
* $H_{\theta}^{(l)}$: 不带 adapter 时，第 $l$ 层输出的节点嵌入矩阵，仅由 GNN 参数 $\theta$ 产生
* $f_{conv, \theta}^{(l)}(\cdot)$: 第 $l$ 层图卷积函数, 用于聚合消息和更新 embeddings 的图卷积函数, 得到该层基础 GNN 表示。
依据图结构 $A$，从邻居节点聚合信息。把上一层嵌入变换成当前层嵌入
* $f_{adap, \theta^\ast}(\cdot)$: 轻量级适配器模块，它可以是任意实现（例如多个全连接层）。它学习一个任务特定的表示修正项（correction term），帮助 unified model 更接近 fine-tuned model 的表示，从而缓解 representation bias
    在公式（4）是 GraphAdapter 函数，也就是图级 adapter 模块。它学习一个图级修正项，用来补偿 unified model 与 fine-tuned model 在最终图表示上的偏差
* $W_{down}$: 降维矩阵（down-projection matrix）
* $W_{up}$: 升维矩阵（up-projection matrix）
* $h$: 图级嵌入向量（final graph embedding vector）
* $h_{\theta, \theta^\ast}$: 带 GraphAdapter 时得到的图级嵌入向量; 这是 经过图级适配修正后的图表示
* $f_r(\cdot)$:  readout function（读出函数），把节点级表示聚合成图级表示。
* $H^{(l)}_{\theta_k}$: 原始微调模型（专家模型）在第 $l$ 层生成的节点嵌入矩阵
* $H^{(l)}_{\theta_{uni},\theta^*_k}$: 统一模型（包含第 $k$ 个任务适配器）在第 $l$ 层生成的节点嵌入矩阵
* $h^{(l)}_i, h'^{(l)}_j$: 分别代表统一模型和微调模型中第 $i$ 或第 $j$ 个节点的嵌入向量
* $TWD(\cdot)$:
* $\theta^\ast_k$L: 第 $k$ 个任务的适配器参数，是task-specific adapter阶段唯一需要优化的变量
* $G \in \mathcal{D_k}$ : 从任务 $k$ 的私有数据集中, 取出输入图样本
* $\mathcal{L_{MD}}$: 图级对齐损失（Manhattan Distance），用来衡量图嵌入向量的差异。
* $\mathcal{L_{TWD}}$: 节点级对齐损失，用来衡量每层节点分布的差异。
* $\alpha$: 平衡系数（超参数），用来控制图级损失 $L_{MD}$ 与 节点级损失 $\sum_l L_{TWD}$ 的权重比例
* $L$：	GNN 模型的总层数





### 4.3 inference procedure


$$f_{moe, \{\theta^*_1, \theta^*_2, ..., \theta^*_K\}}(H) = \sum_{k=1}^K w_k \cdot f_{adap, \theta^*_k}(H) \qquad (7)$$

$$\{w_1, w_2, ..., w_K\} = \text{softmax}(\{-TWD(f_{adap, \theta^*_i}(H), f_{adap, \theta^*_k}(H))\}^K_{i=1}) \qquad (8)$$

$$\{w_1, w_2, ..., w_K\} = \text{softmax}(\{-||f_{adap, \theta^*_i}(h) - f_{adap, \theta^*_k}(h)||_1\}^K_{i=1}) \qquad (9)$$
> 公式（8）：节点级 router，用 TWD 衡量不同 adapter 的节点表示输出是否相似。 
公式
>（9）：图级 router，用 L1 / Manhattan Distance 衡量不同 adapter 的图表示输出是否相似


**公式（7）** ：把多个任务专属 adapter 当成 experts，对同一个输入 $H$, 把 K 个任务专属 adapter 的输出按权重加权求和整合为一个统一的 MoE 适配器模块，根据当前输入图与各任务知识的相似程度，把多个 adapter 的知识 按比例混合 起来，形成一个统一的 MoEAdapter 输出。

**公式（8）** 它通过 TWD 距离来衡量当前任务（任务 $k$）与其他所有任务专家（专家 $i$）的接近程度。如果某个专家 $i$ 的输出与当前任务适配器的输出高度相似，该专家就会被分配更高的权重 $w_i$，从而实现跨任务的知识互补

**公式（9）** 根据 各任务 GraphAdapter 输出与目标任务 GraphAdapter 输出在图级表示上有多接近，动态为每个 expert 分配 graph-level 权重
作者在推理阶段不仅希望在节点级整合多个 task-specific adapter 的知识，也希望在图级整合这些 adapter 的知识。为此，作者先比较 不同任务 adapter 在同一个图级 embedding $h$ 上的输出是否相似，如果相似，则说明这些任务在当前图实例上可能共享有用知识，于是应该给这些任务更大的 MoE 权重。


* $f_{moe, \{\theta^*_1, \theta^*_2, ..., \theta^*_K\}}()$: MoE 适配器的最终输出。代表了所有专家加权协作后的总修正量
* $H$: 输入节点嵌入矩阵。推理时，它是当前层基础模型生成的特征表示
* $\theta_k^\ast$: 第 $k$ 个任务对应的 adapter 参数。它是在 phase (II) 中训练得到的 task-specific adapter 参数。
* $w_k$: 第 k 个 expert 的 路由权重
* $f_{adap, \theta^*_k}(H)$: 第 $k$ 个任务专属 adapter，基于输入 H 计算该任务特有的表示修正项  (矩阵)
* $f_{adap, \theta^*_i}(H)$: 第 $i$ 个任务专属 adapter 作用在节点嵌入矩阵 $H$ 上后的输出 (矩阵)
* $K$: 已有任务/专家的总数量
* $TWD(\cdot)$: 在当前输入图结构下（约束），量化两个离散概率分布之间的差异（即距离）
* $h$: 图级 embedding 向量，这是某个输入图经过 pooling/readout 后得到的整图表示
* $∥\cdots∥_1$: L1 距离
