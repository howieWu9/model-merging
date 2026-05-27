---
title: Exact Upper and Lower Bounds for the Output Distribution of Neural Networks with Random Inputs

---

# Exact Upper and Lower Bounds for the Output Distribution of Neural Networks with Random Inputs

Andrey Kofnov 1 Daniel Kapla 1 Ezio Bartocci 2 Efstathia Bura 1

ICML 2025

源代码：https://github.com/URWI2/Piecewise-Linear-Transformation



## key terms

1. 累计分布函数，**Cumulative Distribution Function**, CDF
    
    描述 一个随机变量取值 小于等于某个数  的概率，通常记作：
    $$F(x)=P(X≤x)$$
    
    其中：
    * X：随机变量
    * x：某个具体的数
    * F(x)：就是当 𝑋 小于等于 𝑥 时的概率

2. 概率密度函数（probability density function，pdf)：
    
    这个函数告诉我们输入数据在空间中是如何分布的（例如，哪些地方的数据出现的概率更高）。

3. $ReLU$ 函数的定义：

$$ReLU(x) = max(0, x) = \cases{x, x \gt 0 \\
0, x \le 0}$$

<<<<<<< Updated upstream

## 观后 问题：
不知道 $\phi(x)$ 这个密度函数具体是这么计算的？文章没给出计算方法，只是假设服从beta分布
=======
>>>>>>> Stashed changes
    
    




## 文章做法

在 输入有随机扰动 的情况下，为 已训练好的神经网络模型（Prediction NN）在面对随机输入时的output，提供精确的数学边界（即累积分布函数 的上下界），而且这个上下界可以不断收紧，最终逼近真实分布




## 2. statement of the problem

> neural network(NN) 的 累计分布函数(cumulative distribution function, cdf)

$$F_{\tilde Y}(y)=P(\tilde Y\le y), \quad where \quad \tilde Y = f_L(X \mid \Theta)$$

* $y$: 拿来评估 cdf 的一个给定阈值点






## 3. approximation 方法

$$\tilde Y = f_L(X \mid \Theta) \quad \text{训练好的神经网络}$$

$$\underline{F}_{\tilde Y}(y)\le F_{\tilde Y}(y)\le \overline{F}_{\tilde Y}(y), \quad \forall y, \quad where \quad F_{\tilde Y}(y)=P(\tilde Y\le y) \tag{4}$$

$$F_{\tilde Y}(y)=\int_{\{\tilde Y\le y\}}\phi(x)\,dx \qquad (5)$$

$$\underline{F}_{\tilde Y}(y) = \int_{\{\tilde Y\le y\}}\underline{\phi}(x)\,dx \le F_{\tilde Y}(y) \le \int_{\{\tilde Y\le y\}}\overline{\phi}(x)\,dx = \overline{F}_{\tilde Y}(y) \qquad (6)$$


* $F_{\tilde Y}(y)$: 一个累计分布函数，输出 神经网络output $\tilde Y$ 取值不超过 $y$ 的概率。






### 3.1 如果网络是ReLU，cdf的计算

$$F_{\tilde Y}(y)= \mathbb{P}[\tilde Y\le y]
=\sum_{i=1}^{q_\phi}\sum_{j=1}^{q_Y} \mathcal{I}[\phi_i(x); \mathcal{P}^r_{j,i}] = \sum_{i=1}^{q_\phi} \sum_{j=1}^{q_Y} \sum_{s=1}^{S_{i,j}} \mathcal{I} [\phi_i(x); \tau_{i,j,s}] ,$$

$$\mathcal{P}_{j,i}^{r} = \mathcal{P}_j \cap k_i \cap \{x : NN^j(x)\le y\} = \bigcup_{s=1}^{S_{i,j}} \tau_{i,j,s} , \qquad (7)$$

$$\{x:NN^j(x)\le y\} = \bigcap_{t=1}^{n_L}\{x:NN_t^j(x) \le y_t\}.$$

> 所有让第 $j$ 个局部网络输出不超过 $y$ 的输入 $x$，本质上是这个输入 $x$ 产生的整个输出向量，整体都不超过阈值向量 $y$。


**公式（5）** 把 **输出分布**问题 转化为 **输入空间积分** 问题。因为，神经网络输出分布不好直接算，但输入分布 $\phi(x)$ 是已知或可设定的。于是作者把问题改写成：在输入空间里找出所有满足 $f_L(x\mid\Theta)\le y$ （第L层的神经网络输出，即最后一层的最终输出）的点, 再对输入密度 $\phi(x)$ 在这个区域上积分

**公式（7）** 文章提出了一个关键假设：输入 pdf $ϕ(x)$ 是一个分段多项式。这意味着整个输入空间并不是由一个统一的数学公式描述的，而是被划分为许多个互不相交的单纯形（Simplex），每一个单纯形记为 $k_i$。

$$\phi(x) = \begin{cases}
\phi_1(x), & x\in k_1^\circ\\
\phi_2(x), & x\in k_2^\circ\\
\vdots\\
\phi_{q_\phi}(x), & x\in k_{q_\phi}^\circ
\end{cases}
$$


* $\tilde Y$: 把随机输入 $X$ 放进已经训练好的神经网络 $f_L$ 后得到的输出
* $y$: 拿来评估 cdf 的一个给定阈值点
* $\{\tilde Y\le y\}$: 所有使得输出 $\tilde Y$ 不超过 $y$ 的输入点所对应的区域
* $x$: 输入空间中的**一个点**，也就是随机输入 $X$ 的一个可能取值
* $\phi(x)$: 随机变量 X (输入)的概率密度函数（probability density function, pdf）
* $\underline{\phi}(x), \overline{\phi}(x)$: 输入随机变量密度 $\phi(x)$ 的 上界 和 下界估计函数
* $q_\phi$:  输入 pdf 被分成的块数总数
* $q_Y$: ReLU 神经网络将 划分为凸多形（Convex Polytopes）的总数量（输入空间切出来的局部线性区域数量）；每个多边形对应一个特定的激活模式
* $\phi_i(x)$:  当 $x$ 落在第 $i$ 块区域 $k_i$ 里面时，$\phi(x)$ 在这块上的具体公式。$\phi(x)$ 在整个输入空间都用同一个公式，而是把输入空间切成很多块 $k_i$。然后在每一块内部，密度函数都由一个多项式来表示 $\phi(x)=\phi_i(x), \quad x\in k_i^\circ$
* $\mathcal{I}[\phi_i(x);P^r_{j,i}]$: 表示 某个函数在某个区域上的积分；所以这个式子表示：把多项式 pdf 片段 $\phi_i(x)$ 在区域 $P^r_{j,i}$ 上做积分
* $\mathcal{P}^r_{j, i}$: 它是 the reduced polytope（约化后的多面体区域）。就是把 **网络第 $j$ 个线性区域**、**pdf 的第 $i$ 个分片区域**、以及 **输出不超过 $y$ 这个条件区域** 三者相交后得到的局部区域
* $S_{i, j}$: 区域 $\mathcal P^r_{j,i}$ 被切分成的 simplex（单纯形）的数量。
* $\tau_{i,j,s}$: 对 $\mathcal P^r_{j,i}$ 做三角剖分后得到的第 s 个 simplex 小区域，用来在上面精确积分。
* $\mathcal{P}_j$: ReLU 网络把 输入空间 切分出来的第 $j$ 个线性区域。
* $k_i$: 输入空间中的第 $i$ 个小几何区域
* $NN^j(x)$: 神经网络在第 $j$ 个线性区域上的局部表达式
* $n_L$: 输出层的维度
* $NN^j(x)$: 神经网络在第 j 个线性区域 $\mathcal P_j$ 上的局部表达式。
    当输入 $x$ 落在某个固定区域（第 $j$ 个区域 $\mathcal P_j$）里面时，这些开关状态不变，于是整个网络的计算路径也固定了。这个网络会退化成一个简单的仿射函数 $f_L(x)=NN^j(x)=c^j+V_jx,\quad x\in \mathcal P_j$。其中 $V_j$ 是整个网络在第 $j$ 个激活区域下，把所有层连起来之后，最终得到的总线性系数矩阵。
* $NN_t^j(x)$: $NN^j(x)$ 这个局部输出向量的第 $t$ 个分量。
* $y_t$: 阈值向量 $y\in\mathbb R^{n_L}$ 的第 t 个分量






### 3.2 构造ReLU上下界网络，对神经网络进行上下近似

$$0 ≤ \tilde Y(x)− \underline Y_n(x) <ϵ, \qquad 0 ≤ \overline Y_n(x)− \tilde Y(x) < ϵ$$

在某个小区间 $[a_k,a_{k+1}]$ 上，如何构造 $\tilde f$ ?

**构造方法有两种**：线性插值法 和 切线拼接法 
针对目标函数的凹凸性交替使用，来构造上下界。

#### 两种构造上下界限近似函数的方法

> 1. **线性插值法** （linear interpolation）

在第 k 个小区间 $[a_k,a_{k+1}]$ 上，用一种叫 linear interpolation (线性插值) 的方法来构造近似函数

用 $a_{k'}$ 左右半段 两条直线拼成一个折线 来逼近上下限函数的局部。如果当前这段是凸函数段，折线插值在上面，做上界；反之，做下界。

(1)
$$a_{k'} = a_{k'}^{lin_int}, \quad a_{k'} \in [a_k,a_{k+1}], \qquad where \quad a_{k′} = \frac{(a_k + a_{k+1})}{2}$$

$$\tilde f(x)=\tilde f^{\text{lin_int}}(x) \quad \text{for}x\in[a_k,a_{k+1}]$$

(2) 弦线
$$\kappa_1 = \frac{f(a_{k'})-f(a_k)}{a_{k'}-a_k}, \qquad \kappa_2 = \frac{f(a_{k+1})-f(a_{k'})}{a_{k+1}-a_{k'}}$$

$$\tilde f(\tau)=f(a_k)+(\tau-a_k)\kappa_1,\qquad \tau\in[a_k,a_{k'}] \qquad \text{左半段折线}$$

$$\tilde f(\tau)=f(a_{k'})+(\tau-a_{k'})\kappa_2, \qquad  \tau\in[a_{k'},a_{k+1}] \qquad \text{右半段折线}$$



* $a_{k'}$: 给 linear interpolation 用的那个中间点
* $\tilde f(x)=\tilde f^{\text{lin_int}}(x)$: 表示在当前这段区间上，把近似函数 $\tilde f$ 定义成这个线性插值函数。
* $\kappa_1$: 这是点 $(a_k,f(a_k))$ 和 $(a_{k'},f(a_{k'}))$ 之间连线的斜率。





> 2. **切线拼接法**

在区间 $[a_k,a_{k+1}]$ 上，也可以用另一种方法，叫 piecewise tangent，来构造近似函数。

(1)
$$a_{k'} = a_{k'}^\text{pie_tan}, \quad a_{k'} \in [a_k,a_{k+1}]$$

$$\tilde f(x)=\tilde f^{\text{pie_tan}}(x)\quad \text{for}x\in[a_k,a_{k+1}]$$

(2) 切线
$$a_{k'} = \frac{ f(a_k)-f(a_{k+1}) - \big(f_+'(a_k)a_k - f_-'(a_{k+1})a_{k+1}\big)}{
f_-'(a_{k+1}) - f_+'(a_k)}$$

$$\tilde f(\tau) = f(a_k)+f_+'(a_k) (\tau-a_k),\qquad \tau \in [a_k,a_{k'}]$$

$$\tilde f(\tau) = f(a_{k+1}) + f_-'(a_{k+1})(\tau-a_{k+1}), \qquad \tau \in [a_{k'}, a_{k+1}]$$

* $a_{k'}$: 在切线拼接方法里，左右两条切线 相交的交点 $(a_{k'},\, \tilde f(a_{k'}))$ 的横坐标
* $\tilde f(\tau)$: 构造的用来近似目标函数的切线

#### 一般的分段线性函数写法

无论是用插值法还是切线法，最后得到的 $\tilde f$ 都是一个分段线性函数(上界/下界)。

$$\tilde f(\tau)=c_i+v_i\tau,\qquad \tau\in[x_{i-1},x_i]$$



#### 把分段线性函数写成 ReLU 形式

$$\tau \in [x_0, x_n], \quad and \quad let \quad v_0 = 0$$

$$\tilde f(\tau) = x_0v_1 + c_1 + \sum_{i=1}^{n} \xi_i  \mathrm{ReLU} \big(|v_i - v_{i-1}| (\tau - x_{i-1})\big), \quad where \quad \xi_i = \mathrm{sign}(v_i - v_{i-1})$$

公式说明了如何计算当前点 $\tau$ 这一段的近似函数值：起点那一段的函数值 + 从起点到当前点 $\tau$ 的累计函数值增量（$\Delta {v_i} \times \Delta {x_i}$）


* $sign()$: 取正负号
* $\mathrm{ReLU}(|v_i-v_{i-1}|(\tau-x_{i-1}))$: 表示从断点 $x_{i-1}$ 开始，触发一个新的线性修正项；意思是函数值改变了多少。




### 3.3 推广到 任何定义在紧致区域上的连续函数

$$\underline F_n(y) \le F(y) \le \overline F_n(y), \quad \forall y \in\{W(X):X \in K\} \quad and \quad \underline F_n(y) → F(y), \overline F_n(y) → F(y) $$

$$\int_{\{W(X)=y\}} \phi(X) dx = 0 \tag{8}$$

公式（8）：为了保证 **uniform convergence**，不能有一批输入 $X$，被函数 $W$ 全部变成同一个输出值 $y$，而且这批输入本身还占了正概率（$P(W(X)=y)>0$）；如果某个点有这种 概率堆积，cdf 在这个点会有跳跃（突然增加一截）。

一旦有跳跃，要让上下界在整个范围内都统一贴近真实 cdf，就更麻烦，所以作者才专门加约束：恰好映到这个值的输入集合概率都必须为 0，以避免出现跳跃情况。
例如：假设输入 $X$ 在 $[0,1]$ 上均匀分布。 $W(X)=
\begin{cases}
0, & x\in[0,0.5]\\
x, & x\in(0.5,1]
\end{cases}$，那么有整整一半输入，被压成了同一个输出值 0，所以 cdf 在 $y=0$ 会突然跳 0.5。



* $y$ 这个阈值取自函数 $W$ 在区域 $K$ 上的值域。
* $X$ 是随机输入向量
* $W$ 是一个从 $K$ 到实数的连续函数
* $\phi(x)$ 是它的概率密度函数，且 $\phi(x)$ 是连续的
* $K$ 是输入空间中的一个紧致超矩形，也就是一个有界闭盒子，比如高维的区间盒
* $\{W(X)=y\}$ 是点的集合







