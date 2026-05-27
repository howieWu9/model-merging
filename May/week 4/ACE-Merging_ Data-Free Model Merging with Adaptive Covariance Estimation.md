---
title: 'ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation'

---

# ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation

CVPR 2026

code: https://github.com/unravel-xu/ACE-Merging/tree/main


## key concepts
* Gram matrix：

    就是一个矩阵和它自己的转置相乘，得到 **列与列之间相似度** 的矩阵。

* 残差（residual）

    指的是 误差向量，即$e = f_\theta(x) - y$
    
* 残差能量 (residual energy)

    它是一个 scalar，指的是误差向量的范数平方 $e^\top e$


* heterogeneity （异质性/非均匀性）：不同对象之间“不一样的程度”。

    指的是 不同任务的 task vector：$\Delta W_1,\Delta W_2,\dots,\Delta W_T$, 它们的大小是不是差很多。
    如果某个任务的更新($\|\Delta W_t\|_F^2$)很大, 另一个任务的更新很小; 直接把它们加起来或者平均，大的任务可能会支配融合结果 (更偏向某个/某些任务)。









## 3.1 model merging objective


$$\min_{\bar{W}} \sum_{t \in [T]}
\mathbb{E}_{x \sim \mathcal{D}_t} \left( 
\left\| f(\bar{W}, x) - f(W_t, x) \right\|_2^2 \right)
\tag{1}$$

* $\bar{W}$: 融合后的模型参数
* $T$: 任务集合



## 3.2 Optimal merging under local linearization

$$f(W,x) \approx Wx, \qquad
\Sigma_t = \mathbb{E}_{x\sim\mathcal{D}_t} [xx^\top]$$

$$L(\bar{W}) = \sum_{t\in[T]} \mathbb{E}_{x\sim\mathcal{D}_t} \left\| (\bar{W}-W_t)x \right\|_2^2 = \sum_{t\in[T]} \operatorname{Tr} \left( (\bar{W}-W_t)\Sigma_t(\bar{W}-W_t)^\top
\right)
\tag{2}$$


$$minimize \quad Eq(2) \to \bar{W} = \left( \sum_{t\in[T]} W_t\Sigma_t \right) \left( \sum_{t\in[T]} \Sigma_t \right)^{-1}
\tag{3}$$


* $Tr(\cdot)$: 矩阵的迹
* $\Sigma_t$: 叫做 任务 $t$ 的输入二阶矩矩阵 




## 4. Methodology

## 4.1 Estimating input second-moment structure

 > 建立输入二阶统计量与微调更新之间的联系

$$\mathbb{E}_{(x,y)\sim \mathcal{D}_t} \left[
g(x,y)^\top g(x,y) \right] \propto \Sigma_t,
\qquad \Sigma_t = \mathbb{E}_{x\sim \mathcal{D}_t} [xx^\top]
\tag{4}$$

根据local linearization (3.2 的第一行公式)，那么每个样本的梯度可以写成

$$g(x,y)=2ex^\top, \quad e=W_0x-y;
\quad \to \quad g(x,y)^\top g(x,y) = 4xe^\top ex^\top$$


$$\mathbb{E}_{(x,y)\sim \mathcal{D}_t}
[g(x,y)^\top g(x,y)] = 4 \mathbb{E}_{(x,y)\sim \mathcal{D}_t} [(e^\top e)xx^\top]$$

如果残差能量与预期中的输入方向近似解耦，那么:

$$\mathbb{E}_{(x,y)\sim \mathcal{D}_t} [(e^\top e)xx^\top] \approx \mathbb{E}_{(x,y)\sim \mathcal{D}_t} [e^\top e] \mathbb{E}_{x\sim \mathcal{D}_t} [xx^\top]$$

$$\mathbb{E}_{(x,y)\sim \mathcal{D}_t} [g(x,y)^\top g(x,y)] \propto  \left(\mathbb{E}_{x\sim \mathcal{D}_t} [xx^\top] = \Sigma_t \right)$$

$$\widetilde{W}_t = \Delta W_t - \mathbf{1}\mu_t^\top,
\qquad
\mu_t = \frac{1}{d_{\text{out}}} \Delta W_t^\top \mathbf{1}
\tag{5}$$

$$\widehat{\Sigma}_t \propto \widetilde{W}_t^\top \widetilde{W}_t = (\Delta W_t-\mathbf{1}\mu_t^\top)^\top
(\Delta W_t-\mathbf{1}\mu_t^\top)
\tag{6}$$


公式（6）就是作者提出的“无数据估计方法”。



* $g(x, y)$: 单样本梯度, 即损失函数对模型权重 $W$ 的梯度。
* $\propto$: 表示 左右两边的东西成比例。
* $\widetilde{W}$: 表示 去掉 $\Delta W_t$ 在 output dimension 上的每列元素平均值之后的 $\Delta W_t$； 文章叫做 centered task vector。
* $\widetilde \Sigma_t$: 任务特定的二阶代理量 (Task-Specific Second-Order Proxy)。原本的 $\Sigma_t$ 需要输入来支持计算，但 它可以直接从 $\Delta W_t$ 中推导出来，并可以当作 $\Sigma_t$ 的一个**估计值**。



## 4.2 Adaptive covariance normalization

> 如果直接把所有 $\widehat{\Sigma}_t$ 加起来，如何避免 大的更新量的 task vector 会主导融合，小更新量的 task vector 会被压制 这个问题？


$$\gamma = \frac{ \operatorname{Var}_t \left[
\log \|\Delta W_t\|_F^2 \right]}{
\left( \mathbb{E}_t \left[ \log \|\Delta W_t\|_F^2
\right] \right)^2}
\tag{7}$$

当检测到强烈的任务异质性($\gamma > \tau$), 按其迹对每个估计器进行归一化：

$$\widehat{\Sigma}_{t,\text{scaled}} = \frac{\widehat{\Sigma}_t}{\operatorname{Tr(\widehat{\Sigma}_t)}}
\tag{8}$$

$$\widehat{\Sigma}_{t,\text{reg}} = \widehat{\Sigma}_{t,\text{scaled}} + \frac{\epsilon}{\operatorname{Tr(\widehat{\Sigma}_t)}}I
\tag{9}$$

对于相对同质的任务集（$\gamma \le \tau$）:
 
$$\widehat{\Sigma}_{t,\text{reg}} = \widehat{\Sigma}_t + \epsilon I
\tag{10}$$



* $\gamma$: 用来衡量 任务间的 task vector 更新量大小差异。
* $\operatorname{Var}_t [\cdot]$: variance operator
* $\epsilon$: 正则化强度




## 4.3 Collective structural prior

$$\widetilde{\Sigma}_t = \begin{cases}
\widehat{\Sigma}_{t,\text{scaled}}, & \gamma > \tau \\
\widehat{\Sigma}_t, & \gamma \le \tau
\end{cases}
\tag{15}$$

$$c^\top = \frac{1}{d_{\text{in}}} \mathbf{1}^\top
\sum_{t\in[T]} \widetilde{\Sigma}_t,
\qquad
C_{\text{agg}} = \mathbf{1}c^\top
\tag{11}$$

在异质状态（$\gamma > \tau$）中，进一步根据原始任务向量的平均能量对该先验进行重新缩放：

$$\bar{\tau} = \frac{1}{T} \sum_{t\in[T]}
\operatorname{Tr} \left( \Delta W_t^\top \Delta W_t \right) = \frac{1}{T} \sum_{t\in[T]}
\|\Delta W_t\|_F^2
\tag{12}$$

$$C_{\text{agg}} \leftarrow \frac{C_{\text{agg}}}{\bar{\tau}}
\tag{13}$$

得到 preliminary merged update:

$$\bar{W}_{\text{pre}} = \left( \sum_{t\in[T]}
\widetilde{W}_t \widehat{\Sigma}_{t,\text{reg}} \right) \left( \sum_{t\in[T]} \widehat{\Sigma}_{t,\text{reg}} + C_{\text{agg}}
\right)^{-1}
\tag{14}$$


* $\bar{\tau}$: 平均能量尺度，代表了所有参与合并的任务向量在权重空间位移的平均强度




## 4.4 谱精修

在任务异质性很强时 ($\gamma > \tau$ 时)，修正 $\bar{W}_{\text{pre}}$  的奇异值分布，避免能量过度集中在少数方向，需要如下的改进：

$$\widetilde{\Sigma}_t = \begin{cases} \widehat{\Sigma}_{t,\text{scaled}}, & \gamma > \tau \\
\widehat{\Sigma}_t, & \gamma \le \tau
\end{cases}
\tag{15}$$

先计算平均结构：

$$\Sigma_{\text{reg}} = \frac{1}{T} \sum_{t\in[T]} \widehat{\Sigma}_{t,\text{reg}}
\tag{16}$$

用 residual 找回被平均过程压掉的结构信息：

$$\Delta_{\text{res}} = \sum_{t\in[T]} \Delta W_t
\left( \widetilde{\Sigma}_t - \bar \Sigma_{\text{reg}} \right)
\tag{17}$$

$$\bar{W}_{\text{fused}} = \bar{W}_{\text{pre}} + \Delta_{\text{res}}$$

$$\bar{W}_{\text{fused}} = USV^\top,
\qquad
S=\operatorname{diag}(\sigma_1,\sigma_2,\dots)$$


$$\Delta W_{\text{refine}} = \sigma_{\text{iso}} U_{:,1:k} V_{:,1:k}^\top,
\qquad
\sigma_{\text{iso}} = \frac{1}{k} \sum_{i=1}^{k} \sigma_i
\tag{18}$$


$$\bar{W} = \bar{W}_{\text{pre}} + \Delta W_{\text{refine}}
\tag{19}$$




公式（17）：对每个任务，计算 第 $t$ 个任务的结构 与 平均结构之间的差异 (任务 $t$ 相比于其他任务，具有哪些独特的方向)，然后把差异重新注入到 task update 里。最后求和得到 所有任务共同的结构补偿项。

公式（18）： 保留 $\bar{W}_{\text{fused}}$ 子空间的主要方向，同时压制 top-k 个方向的奇异值，避免少数最大奇异值过大，导致能量过度集中。


* $U_{:,1:k}$: 取 U 的前 k 列
* $V_{:,1:k}$: 取 V 的前 k 列



