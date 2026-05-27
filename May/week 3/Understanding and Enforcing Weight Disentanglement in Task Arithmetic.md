---
title: Understanding and Enforcing Weight Disentanglement in Task Arithmetic

---

# Understanding and Enforcing Weight Disentanglement in Task Arithmetic

CVPR 2026

https://github.com/RL-MIND/OrthoReg




## 对我们文章的帮助

在融合前，选择彼此更正交、干扰更小的 expert/model。 -> 公式（8）





## 3. preliminaries 

### 3.2 Weight Disentanglement
$$\begin{aligned}
f\left(x; \theta_0 + \sum_{t=1}^{T} \alpha_t \tau_t\right) =
\begin{cases}
f(x; \theta_0 + \alpha_i \tau_i), & \text{if } x \in \mathcal{D}_i \\
f(x; \theta_0), & \text{if } x \notin \bigcup_{t=1}^{T} \mathcal{D}_t
\end{cases}
\end{aligned} \tag{3}$$

合并多个 task vector 后，某个任务的数据应该只受到它自己 data domain 的 task vector 影响，不应该被其他任务干扰。 



### 3.3 The NTK Linearization Hypothesis

$$f(x; \theta_0 + \tau) \approx f(x; \theta_0) + \tau^{\top}\nabla_{\theta} f(x; \theta_0) \tag{4}$$








## 4. Proposed Framework

### 4.1 An Equivalent Condition for Disentanglement

只考虑 i 和 j 两个任务的情况：

$$f(x;\theta_0+\tau_t+\tau_j) = f(x;\theta_0+\tau_t),
\qquad \forall x\in \mathcal{D}_t \tag{5}$$

(5): 对于任务 $t$ 的数据 $x$，如果把任务 $t$ 的 task vector 和任务 $j$ 的 task vector 都加到模型里, 模型输出应该和 只加任务 $t$ 的 task vector 的输出一样。


也就是说，对($\tau_t + \tau_j$)展开中：
$$\tau_j^{\top} J(x) = 0, \quad \forall x \in \mathcal{D}_t , \qquad J(x) := \nabla_{\theta} f(x;\theta_0) \tag{6}$$






### 4.2.1 Task-Feature Specialization (TFS)

一个 linear layer 的权重矩阵：$W = \{ w_k \}_{k=1}^d$, 每一个列向量 $w_k$ 看成一个 base feature，也就是一个基础特征提取器。对应地，每个 feature 会产生一个 activation：$z_k$

$$I_t \subseteq \{1,\ldots,d\}$$

$$\mathbb{E}_{x\sim \mathcal{D}_t} \left[ \left| \frac{\partial f(x;\theta_0)}{\partial z_k} \right| \right] =0,
\qquad \forall k\notin I_t$$

$$I_t \cap I_j = \emptyset$$

**如果不同任务依赖的内部特征集合不重叠，那么任务之间就不会互相干扰。**

* $I_t$: 任务 t 使用的 feature 集合, 集合里是任务 t 使用的特征列向量的编号





### 4.2.3 From TFS to Weight Vector Orthogonality

$$I_t \cap I_j = \emptyset, \quad then: \quad \langle w_k,w_l\rangle = 0, \quad \forall k\in I_t,\ l\in I_j,\ t\neq j$$

* $\langle \cdot,\cdot\rangle$: 算 inner product / dot product。





### 4.3.1. The Challenge of Feature Overlap

理想情况下，没有特征重叠。 现实情况是有特征重叠的 $I_t \cap I_j \neq \emptyset$:

$$k\in I_t\cap I_j, \quad x\in\mathcal{D}_t \to \nabla_{w_k}f(x;\theta_0)\neq 0 \to (\tau_j)_k\neq 0$$

$$\text{task interference: } \quad \left\langle (\tau_j)_k,\nabla_{w_k}f(x;\theta_0)\right\rangle \neq 0$$

因此，不能只依赖预训练模型本身的 TFS，而要在 fine-tuning 时主动构造更好的 task vector。





### 4.3.2 Method: Orthogonal Regularizations

$$\mathcal{L} = \mathcal{L}_{\text{task}}(\theta_0+\Delta\theta) + \lambda\cdot\mathcal{L}_{\text{ortho}}(\Delta\theta) \tag{7}$$

$$\mathcal{L}_{\text{ortho}}(\Delta\theta) = \sum_l \left\| (\Delta W^{(l)})^\top \Delta W^{(l)} - I \right\|_F^2 \tag{8}$$

(8): 列向量之间尽量正交，并且长度接近 1。

* $l$: 第 $l$ 个 linear layer。 



### 4.3.3. Theoretical Justification to OrthoReg

$$|\tau_j^\top J(x)| \approx \|\tau_j\|_2\cdot \|J(x)\|_2\cdot |\cos\angle(\tau_j,\tau_t)| \tag{9}$$

参数更新幅度 $\times$ 模型输出对参数变化的敏感程度 $\times$ 两向量余弦值

