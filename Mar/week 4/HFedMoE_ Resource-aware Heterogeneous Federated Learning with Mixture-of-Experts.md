---
title: 'HFedMoE: Resource-aware Heterogeneous Federated Learning with Mixture-of-Experts'

---

# HFedMoE: Resource-aware Heterogeneous Federated Learning with Mixture-of-Experts
Jan 2026
> Zihan Fang, Zheng Lin, Senkang Hu, Yanan Ma, Yihang Tao, Yiqin Deng, Xianhao Chen, Yuguang Fang

没有给出GitHub仓库地址

这份研究目的是解决在大语言模型微调过程中，移动设备计算资源受限与数据隐私保护之间的矛盾。
在不改变总的模型规模情况下，通过引入混合专家模型（MoE）结构 和 information bottleneck理论（根据设备预算动态选择最具贡献的专家） 来降低client的算力消耗。

## key concepts
![image](https://hackmd.io/_uploads/BJqZvGZsWx.png)


## **文章提出的问题：**
> **Figure 2**
>
>这些观察结果强调了识别专家重要性并针对每个客户端选择性地激活专家的必要性，以提高效率和个性化程度。
>
> **Figure 2 (a):**
* 在同一个任务（AGNews）下，Client 1 和 Client 2 在不同层上激活的专家是不一样的
* 因为每个客户端的数据分布不同，所以：专家的重要性是 client-specific（客户端相关的），不能假设一个统一的全局路由策略适合所有客户端

**Figure 2 (b):**
* gating network 被聚合以后，global router 倾向于保留 所有client上都常用 的专家（没有针对性）
* 专家数量多，不代表选得对；在联邦异构数据下，选对专家比多激活专家更重要（有针对性的为每个client做选择）

> **Figure 3**
> 
> 在不同批次大小下激活专家比例以及在不同训练失败率下的收敛性能（训练轮次和测试准确率）。
> 
> Figure 3(a) 说明：batch 一大，MoE 在 batch 级别会同时激活越来越多专家，导致资源消耗升高；Figure 3(b) 说明：这种资源压力会让算力弱的客户端更容易训练失败，而更高的训练失败率会进一步导致联邦学习收敛更慢、精度更差。
**Figure 3 （a）**
* 虽然 MoE 是 top-1/top-k 按 token 选专家，但从 整个 batch 来看：随着 batch size 变大，一个 batch 内不同 token 会路由到更多不同的专家，因此每层 累计被激活的专家比例 会上升

**Figure 3 （b）**
* 算力有限的客户端会频繁无法完成本地 LLM 微调，而随着训练失败率升高，模型收敛所需的训练轮数增加，而最终准确率下降

> **Figure 4**
> 第三个挑战：在 MoE-based FL 里，不同客户端只训练了不同 experts、而且路由偏好也不同
> 
> 因此统一平均这种做法会产生聚合失配。
> Figure 4(a) 显示了某层的某个expert被多少个client选中了，用 expert 激活差异证明“客户端更新的不是同一个子模型”
> Figure 4(b) 每个client在聚合后都发生了accuracy下降的问题，因为整个 MoE 模型被按普通 FL 的方式统一聚合所导致
> 
> 从而促使我们设计一种资源感知的专家选择机制

> **Figure 5**
> a): 在当前 batch 内给每个 expert 打分，计算重要程度，根据：
> Cumulative Importance（累计重要性）对应论文公式 (3)
> Specific Importance（特定重要性）对应论文公式 (4)
> 
> b): 
> expert selection modeling：根据得到的 **importance score** ，在预算内，从触发的 experts 中选出最值得训练的几个 experts
> 
> IB-Guided re-scheduling：


> **Figure 6**
> a): 减少专家数量的同时，计算成本 和 性能都在显著下降
> 
> b):  专家选择高度不平衡，其中一小部分专家贡献了大部分性能提升，而其他专家的贡献微乎其微。
> 作者通过a 和 b 的现象，提出了 基于 **information bottleneck** 原理的专家选择建模



## 文章的方法
### Expert Importance Identification
> $$𝓔^c = \text{Top-k} \left(\left\{\operatorname{Softmax}(G(x))\right\}_{e\in 𝓔}\right), \quad (1)$$
> 
> $$y = \sum_{e \in 𝓔^c} G_e(x) \cdot E_e(x), \quad (2)$$
> 
> $$s^{cumul}_b(e) = \frac{1}{B}\sum_{i=1}^{B} G_e(x_i) \quad (3)$$
> 
> $$s^{specific}_{b}(e)=\max_{1 \le i \le B} G_e(x_i) \quad (4)$$
> 
> $$s_b(e)=\lambda \cdot s_b^{\text{cumul}}(e) + (1-\lambda)\cdot s_b^{\text{specific}}(e) \quad (5)$$



（1）在一个 MoE 层中，根据 gating network（门控网络）对输入 $x$ 计算出的路由分数 $G(x)$，归一化后，从全部专家里选出最重要的 top-k 个专家，作为当前 token 要激活的专家集合 $𝓔^c$
(2) 被选中的专家输出 的加权求和，权重来自 gating network 给出的路由分数
(3) 论文定义的 cumulative importance（累计重要性），衡量某个专家 $e$ 在一个 mini-batch 所有样本里内 总体上有多重要（平均被路由到的程度有多高）

(4) 使用第 e 个专家在小批量所有样本中的最大路由得分来捕捉其对任何单个样本的最大相关性和峰值影响
用来保留 低频但高价值 的专家（平时不常被激活，但一旦遇到某类特殊样本 / 难样本 / 少数类样本，就会特别重要）

(5) 把某个 expert（e） 的“通用重要性”和“特定样本重要性”加权合并，得到这个 expert 在当前 batch 中的总体重要性分数，目的是 在 泛化能力（generalization） 和 特异性（specificity） 之间做平衡

* $c$: 客户端（client）的编号 $c \in \{1, \cdots, C \}$
* $Softmax(G(x))$: 对 gating network 给出的专家打分做 softmax 归一化，变成一个 对所有专家的相对权重分布
* $𝓔$: 全部experts的集合 $𝓔= \{E_{1,1},...,E_{L,S} \}$
* $e$: 某个专家的索引
* $s^{cumul}_{b}(e)$: 专家 $e$ 在 第 $b$ 个 batch 上的累计重要性分数
* $b$: 第 b 个 batch（索引）
* $B$: batch size
* $s^{specific}_b(e)$: 在 batch b 中，专家 e 的特定重要性（specific importance）
* $\lambda \in [0, 1]$: 超参数（hyperparameter），用来控制 cumulative importance 和 specific importance 的相对权重


### Resource-aware Expert Selection

> $$\max_{p(z\mid x)} \; I(z;y) - \beta  \cdot I(z;x), \quad z = \{ z_e, ∀e \in  𝓔\} \quad and \quad z_e \in \{0, 1\}  \quad (6)$$
> 
> $$I(z;x) \le \mathbb{E}_x [KL(p(z|x)||p(z))] \quad (7)$$
> 
> $$I(z; y) \ge \mathbb{E}_{(x,y)} [\mathbb{E}_{z \sim p(z|x)}(\log q(y|z))] + H(y) \quad (8)$$
> 
> $$I_b(e)=I(z_e;y)-\beta I(z_e;x)\ge \mathbb{E}_{z_e\sim p(z_e\mid x)}\!\left[\log q(y\mid z_e)\right] + H(y)-\beta \cdot \mathrm{KL}\!\big(p(z_e\mid x)\,\|\,p(z_e)\big) \quad (9)$$

(6) 在有限计算资源下，寻找一种专家激活策略 $p(z\mid x)$，让被选中的专家既尽可能保留与 任务相关的信息，又尽可能减少对输入的冗余依赖

(7)互信息 $I(z;x)$ 通常在计算上是不可行的（intractable）。公式 (7) 通过 KL 散度 为 $I(z;x)$ 提供了一个变分上界
* 公式 (7) 衡量的是：给定输入 $x$ 时的专家激活分布 $p(z\mid x)$ 与一个先验分布 $p(z)$ 相差多大。
* KL散度越小代表两个分布越接近，也就是说：
    1. 在看到特定输入 x 后，模型给出的专家选择策略 ($p(z \mid x)$) 
    2. 简单的、与输入 x 无关的分布  ( $p(z)$ )
* 1. 和 2. 的分布接近，意味着模型刻意忽视了 x 中的大部分细节，使得最终选择专家的决策看起来就像是随机从先验分布中抽取的一样

* 所以，差得越大，说明 $z$ 对输入 $x$ 记得越多，也就意味着压缩得不够、计算冗余更高。

(8) 为互信息 $I(z;y)$ 提供了一个变分下界。它的作用是确保潜变量 z（专家激活模式）能够保留尽可能多关于目标标签 y 的任务相关信息。最大化这个下界等同于 通过数学约束，强制要求模型选择的 专家激活模式 z  必须包含足够的信息来还原出正确的标签 y

(9) 用于评估单个专家 e 的信息瓶颈（IB）贡献度，用来判断该专家是否值得在资源受限条件下被优先激活和训练

* $z$: 一个只有0/1的向量，表示一个batch内的专家激活模式（expert activation pattern），也就是当前输入最终激活了哪些 experts
* $z_e$: z 的第 e 个分量
* $p(z \mid x)$:给定输入 x 时，专家激活模式 z 的条件分布, 也就是说，对于某个输入 x，系统以什么概率激活哪些 experts
* $p(z)$: z 的先验分布（Prior Distribution），文中假设其服从标准正态分布 $N(0,I)$
* $max_{p(z \mid x)}$: 在所有可能的条件分布 $p(z\mid x)$ 中，寻找使目标函数最大的那个选择策略
* $I(z;y)$: z 与 y 的互信息（mutual information），表示expert activation pattern z 保留了多少与任务标签 y 相关的信息。值越大，说明选出来的 experts 对最终任务越有帮助。
* $I(z;x)$: 衡量整个 $z$ 对输入 x 的依赖程度，表示输入信息冗余（redundancy of input information）。值越大，说明激活模式 z 还保留了很多关于输入 x 的信息，表示压缩不够、冗余较大（z里的专家过多）
* * $I(z_e;x)$: 代表了特定专家 e (单个expert) 所携带的输入冗余信息
* $β$：信息瓶颈中的权衡系数（trade-off hyperparameter），用来平衡 task-relevant informativeness 和 resource compression
* $\mathbb{E}[\cdot]$: 期望算子, 计算后面括号内数值的期望值（即加权平均值）
* $\mathbb{E}_x[⋅]$: 对输入 x 的概率分布求期望，计算在所有可能输入下的平均 KL 散度。
* $\mathbb{E}_{(x, y)}[⋅]$: 对 样本-标签对 $(x,y)$  的联合分布取期望
* $z∼p(z∣x)$: 表示随机变量 $z$ 是从条件分布 $p(z\mid x)$ 中采样得到的
* $\mathbb{E}_{z∼p(z∣x)}[\cdot]$: 评估在当前路由决策下，预测结果的平均准确性。
* $\operatorname{log} q(y∣z)$: 给定 z  后标签 y 的对数似然 
* $q(y∣z)$: 一个变分解码器（variational decoder），在已知专家 $e$ 的激活状态 $z_e$ 时，预测出正确标签 $y$ 的对数似然（Log-likelihood）
    * 衡量的是专家 $e$ 对预测结果的贡献度（任务相关性）
    * 编码过程：模型将复杂的输入 x 映射（压缩）为专家激活模式 z。在这里，z 就是输入信息的某种“潜变量表示”
    * 解码过程：为了完成任务，模型需要从这个压缩后的 z 中提取信息来预测标签 y。在这个环节，$q(y∣z)$ 的作用就是将 z 还原（解码）回任务空间 y
    * 因此，本文的理论模型里，$p(z∣x)$ 被视为编码器，而 $q(y∣z)$ 被视为解码器
* $H(y)$: 标签 $y$ 的熵（entropy）
* $I_b(e)$: 专家 e 在 batch b 上的 IB contribution（信息量）
* $I(z_e; y)$: 专家 e 的激活状态 $z_e$ 中到底保留了多少与任务标签 y 有关的信息(衡量专家激活 $z_e$ 与标签 y 的相关性)
* $I(z_e; x)$: 专家 e 的激活状态 $z_e$ 对输入 x 记住了多少冗余信息(衡量专家激活 $z_e$ 对输入 x 的冗余度)



> $$p(z_e\mid x) = G_e(x), \qquad
p(z_e) = \frac{1}{B}\sum_{j=1}^{B} G_e(x_j) \qquad (10)$$
> 
> $$\mathrm{KL}\!\big(p(z_e\mid x)\,\|\,p(z_e)\big)
=
\frac{1}{B}\sum_{i=1}^{B} G_e(x_i)\log \frac{G_e(x_i)}{p(z_e)} \qquad (11)$$
>
> $$\mathbb{E}_{z_e\sim p(z_e\mid x)}\!\left[\log q(y\mid z_e)\right]
=
p(z_e\mid x)\cdot \log q(y\mid z_e) \qquad (12)$$
>
> $$I(z_e;y) \approx f(\{G_e(x)\}_{i=1}^{B}) \qquad (13)$$
> 
> $$I_b(e)=s_b(e)-\beta\cdot \frac{1}{B}\sum_{i=1}^{B}G_e(x_i)\log\frac{G_e(x_i)}{p(z_e)} \qquad (14)$$


(10): 把 给定输入时专家 e 被激活的条件分布 近似成 router 给这个专家的分数。
把大小为 B 的 batch 内所有样本对专家 e 的路由分数求平均，用它近似专家 e 的边缘激活概率

(11): 衡量专家 $e$ 的激活是否 过于输入依赖 
如果某些样本 $x_i$ 上 $G_e(x_i)$  显著偏离 batch 平均的  $p(z_e)$，那么这个 KL 就会变大。
这意味着专家 $e$ 的激活模式对输入非常敏感，保留了更多输入细节，也就意味着更高的冗余或更差的压缩性。

(12): 激活概率 × 专家e对标签的贡献度。如果专家 e 更可能被激活，并且一旦激活就更能解释标签 y，那么它的任务相关收益就更高。

(13): 通过分析专家 $e$ 的 routing behavior，把理论上难以直接计算的 $I(z_e;y)$（专家激活与标签的互信息），转化为一个由 batch 内路由分数集合 $\{G_e(x_i)\}_{i=1}^{B}$ 决定的函数 $f(\cdot)$

(14): 给单个专家 $e$ 在 batch $b$ 上定义一个 **最终可计算的** IB contribution 分数



### IB-Guided Expert Re-scheduling

> $$𝓔_{\mathrm{active}}(c)
=
\arg\max_{\substack{
𝓔_{\mathrm{active}}(c)\subseteq 𝓔_{\mathrm{union}}(c) \\
|𝓔_{\mathrm{active}}(c)| \le C_{\mathrm{budget}}(c)
}}
\sum_{e \in 𝓔_{\mathrm{active}}(c)} I_b(e) \qquad (15)$$

(15): 在客户端 c 的计算预算限制下，从当前 batch 中所有 候选专家 里，选出一组总信息价值最高的专家，让它们参与反向传播

* $𝓔_{active}(c)$:  客户端 c 在当前 batch 上 最终被激活、允许参与反向传播的专家集合
* $arg \max$: 使目标函数取得最大值的 自变量(是谁让这个值最大)
* $𝓔_{union}(c)$: 当前 batch 中，客户端 c 在所有样本上的路由专家并集, 也就是 batch 里所有样本前向传播时曾被选中过的专家总集合
* $C_{budget}(c)$: 客户端在一个 batch 中最多允许激活专家的 数量


### Sparsity-aware Model Aggregation

> $$u_c(e) = \frac{1}{|D_c|} \sum_{x_i \in D_c} G_e(x_i) \qquad (16)$$
> 
>$$E_e^{t+1} = \sum_{c = 1}^{C}\frac{|D_c| \mathbb{𝟏}(u_j(e) \ge \tau)}{\sum_{j=1}^C |D_j| \mathbb{𝟏}(u_j(e) \ge \tau)} E_e^t(c) \cdot 𝟏(u_c(e) \ge \tau) \qquad (17)$$

(16): 公式 (16) 计算每个专家在某个客户端上的 normalized usage $u_c(e)$，作为“这个专家在这个客户端上是否被真正训练过”的证据（越多样本使用了专家e， 那么 $u_c(e)$ 必然越大）

(17): 只有当某客户端对专家 e 的 usage $u_c(e)$ 达到阈值 $\tau$ 时，这个客户端的 $E_e^t(c)$ 才能参与全局聚合

* $u_c(e)$: 专家 e 在客户端 c 上的 normalized usage（归一化使用率）
* $D_c$: 客户端 c 的本地数据集
* $\mid D_c \mid$: 客户端 c 的本地数据集的样本数量
* $E_e^{t+1}$: 全局模型中专家 e 在第 t+1 轮聚合后的参数
* $C$: 参与联邦训练的客户端总数
* $1(u_c(e)≥τ)$: 指示函数（indicator function），判断客户端 c 上专家 e 的 usage 是否达到阈值 $\tau$ ?
* $\tau$: usage threshold（使用率阈值），一个超参数
* $E_e^t(c)$: 在第 t 轮局部微调后，客户端 c 上专家 e 的参数




### Importance-Weighted Gating Aggregation (重要性加权门控聚集)

> $$r(c) = \frac{|\mathcal{S}_c \cap \mathcal{S}_{global}|}{|\mathcal{S}_{global}|}, \quad where \quad \mathcal{S}_c = \{e \mid u_c(e) \ge \tau \} (18)$$
> 
> $$p_c(e) = u_c(e) \cdot s_b^c(e), \quad \forall e \in S_c \qquad (19)$$
> 
> $$\alpha(c) = \frac{\sum_{e \in S_c} p_c(e)}{|\mathcal{S}_{global}|} \qquad (20)$$
> 
> $$G_{\mathrm{gating}}^{t+1} = \sum_{c=1}^{C} \alpha(c) \cdot G_c^t \qquad (21)$$


(18): 衡量 客户端 c 的主导专家集合 $S_c$ 与 全局主导专家集合 $S_{global}$ 的重合程度
数值越高表示聚合路由行为越一致、越可靠

这篇论文把 $r(c)$ 当成“解释路由一致性”的中间概念提出来了，但在真正的聚合权重公式里没有把它作为独立变量显式乘进去；它的作用更像是被 $S_c$、$S_{global}$ 和专家偏好求和项隐式体现了。

(19): 专家 e 在客户端 c 上被使用的频率 $\times$ 专家e 在客户端 c 的本地数据分布下的重要性分数
在客户端 c 看来，专家 e 到底有多重要、多值得在 gating 聚合中被重视

(20): 客户端 c 在全局 gating 聚合中该占多大权重 $\alpha(c)$，这个权重取决于它“偏好的主导专家”总体有多重要
该权重强调了那些既被广泛选择又在本地表现重要的专家所对应的客户端

(21):使用公式 (20) 得到的客户端权重 $\alpha(c)$，对所有客户端的 gating network 参数做加权平均，从而得到下一轮的全局 gating network 参数


* $r(c)$: 客户端 c 的 routing consistency（路由一致性）
* $\mathcal{S}_c$: 客户端 c 的 dominant expert subset（主导专家子集）, 即在该客户端上 usage 达到阈值 $\tau$ 的专家集合
* $\mathcal{S}_{global}$: 在全局范围内 被至少一个客户端认为是主导专家 的专家集合
* $p_c(e)$: 客户端 c 对专家 e 的 expert preference（专家偏好）
* $G_{gating}^{t + 1}$: 第 t+1 轮的 全局 gating network 参数
* $G_c^t$: 客户端 c 在第 t 轮本地训练完成后的 gating network 参数
