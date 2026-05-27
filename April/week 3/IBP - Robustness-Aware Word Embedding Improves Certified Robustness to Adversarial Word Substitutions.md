---
title: IBP - Robustness-Aware Word Embedding Improves Certified Robustness to Adversarial Word Substitutions

---

# Robustness-Aware Word Embedding Improves Certified Robustness to Adversarial Word Substitutions

Yibin Wang, Yichen Yang, Di He, Kun He

ACL 2023

code: https://github.com/JHL-HUST/EIBC-IBP/


## concepts
1. Logits & activations 区别？
2. vector of activations ~ output(=Logits)
3. one-hot vector 就是在某些分量为1其余为0的向量。这样子就可以提取另一个向量中某一个分量的值，或者某些分量的和



## preliminary

$$B_{adv}(x) = \{ \langle x'_1, x'_2, \dots, x'_N \rangle, x'_i \in S(x_i) \cup \{x_i\} \} \tag{1}$$

$$\forall x' \in B_{adv}(x), \qquad f(x') = f(x) = y \tag{2}$$

$$\underline{z}^K_{y_{max}} \ge \overline{z}^K_y, \qquad \forall y \in \mathcal{Y}, y \neq y_{max} \tag{3}$$

$$z^k = f_k(z^{k-1}), \qquad k = 1, \dots, K \tag{4}$$

$$\underline{z}^0_{ij} = \min_{x_i \in S(x_i) \cup \{x_i\}} \phi(x_i)_j, \qquad \overline{z}^0_{ij} = \max_{x_i \in S(x_i) \cup \{x_i\}} \phi(x_i)_j \tag{5}$$

$$\underline{z}^k_i = \min_{\underline{z}^{k-1} \le z^{k-1} \le \overline{z}^{k-1}} e_i^\top f_k(z^{k-1}), \qquad \overline{z}^k_i = \max_{\underline{z}^{k-1} \le z^{k-1} \le \overline{z}^{k-1}} e_i^\top f_k(z^{k-1}) \tag{6}$$


公式(1): 定义攻击模型（threat model），攻击者被允许做的是：对句子中任意位置的词；用该词的同义词替换；或者保持不变。

公式(2): 对于原始输入 $x$ 的所有合法对抗样本 $x'$，模型在这些样本上的预测结果都和原句 $x$ 的预测一样，且这个共同的预测标签是 $y$

公式(3): 即使考虑所有允许的同义词替换扰动，未扰动样本的目标类别 y_{\max} 的 logit 在最坏情况下的最小值，仍然不小于其他任何类别 logit 在最好情况下的最大值。

公式(5): 对于句子中第 $i$ 个位置的词，把它所有允许替换的词（同义词加原词）都映射成 embedding；然后看这个word vector第 $j$ 个分量上，这些词的取值最小是多少、最大是多少；这两个值就构成输入层在该位置该维度上的下界和上界。
因为这样子就可以把 第 i 个词可以替换成哪些词 这种离散不确定性，转成 embedding 空间中 第 j 维可能落在哪个区间 的连续上下界；为后续的 Interval Bound Propagation 提供了输入层的初始 bounds。

公式（6）：假设第 $k-1$ 层的输入不是一个固定向量，而是在某个区间内变化；那么把所有这些可能输入都送进第 $k$ 层函数 $f_k$ 后，第 $i$ 个输出分量能取得的最小值，就是第 $k$ 层第 $i$ 个神经元的下界；能取得的最大值，就是它的上界。

* $x$: 原始输入文本, 一个由 $𝑁$ 个词组成的文本序列
* $x_i'$: 用来替换 第 $i$ 个原始单词 的候选词。
* $S(x_i)$: 词 x_i 的 synonym set
* $S(x_i)\cup\{x_i\}$: 同义词集合 再加上 原词自己；也就是说，扰动文本的词可以(攻击时不是强制每个词都必须改掉，而是可以改，也可以不改)
* $f(X) \quad f : X \to Y$
* $\underline{z}^{K}, \overline{z}^{K}$: 输出层 logits 的下界 和上界 向量
* $y_{\max}$: 模型对干净（未扰动）样本预测出的具有最大 Logit 值的类别索引，表示 输出中最大的那个 logit 对应的类。$y$ 是一个遍历变量，表示任意一个类别
* $\underline{z}^{K}_{y_{\max}}$: 预测类别的输出下界。指模型对原始输入预测出的类别 $y_{\max}$ 在扰动空间内的最小 Logit 值
* $\overline{z}^{K}_{y}$: 非预测类别的输出上界。指除了预测类别以外的任意类别 $y$ 在扰动空间内可能达到的最大 Logit 值
* $\mathcal{Y}$: 输出空间集合
* $z^k$: 第 $k$ 层的激活向量（Activation Vector）。它表示神经网络第 $k$ 层在接收上一层输入并经过处理后的输出。
* $z^0_{ij}$ 表示第 0 层，第 i 个词 $x_i$ 的 word vector的 第 j 个分量
* $\phi(x_i)_j$ 是词 $x_i$ 的 word vector 的第 $j$ 个元素。
* $\phi(x_i) \in \mathbb{R}^D$:  $\phi$ 把一个词映射到 D 维词向量。
* $\overline{z}^{k}_{i}$ : 第 $k$ 层的输出向量 里的第 $i$ 个输出分量
* $e_i$ 是一个单位独热向量 (One-hot Vector)，在第 $i$ 个位置为 1。


## 文章方法
## 4.1 

$$\max(\cdot)\quad \& \quad |\cdot| \quad \text{are element-wise operators}$$


$$\text{minimize } \overline{z}^K_y - \underline{z}^K_{y_{max}}, \qquad \forall y \in \mathcal{Y}, y \neq y_{max} \tag{7}$$

$$\text{minimize } \max_{x_i \in \mathbf{X}} \left( \max_{x'_i \in S(x_i)} \left( \lvert φ(x_i) - φ(x'_i) \rvert \right) \right) \tag{8}$$

公式（8）：对输入句子中的每个词 $x_i$，看它和所有同义词 $x_i'$ 在 embedding 空间中的差距；再取这些差距中的最大者；然后希望把这个 最大差距 最小化。


## 4.2

$$\max(\cdot)\quad \& \quad |\cdot| \quad \text{are element-wise operators}$$

$$d_{bound}(x_i, S(x_i)) = \left\| \max_{x'_i \in S(x_i)} | φ    (x_i) - φ(x'_i) | \right\|_p \tag{9}$$ 

$$d(x_a, x_b) = \| \phi(x_a) - \phi(x_b) \|_p \tag{10}$$

$$\mathcal{L}_{EIBC}(x_i, S(x_i), \mathcal{N}(M)) = d_{bound}(x_i, S(x_i)) - \frac{1}{M} \sum_{\tilde{x}_i \in \mathcal{N}(M)} min(d(x_i, \tilde{x}_i), \alpha) + \alpha \tag{11}$$

公式（9）: 对于某个词 $x_i$，先看它和每个同义词 $x_i'$ 在 embedding 空间中每一个分量相差多少，**得到一堆向量差**；

然后按照逐个维度比较所有向量差，**取出每一维度的最大差值**，得到一个 最坏情况差异向量；

最后再对这个向量取 p-norm，得到一个标量，作为该词同义词集合在 embedding 空间中的 bound 大小。

公式（10）:两个词 $x_a$ 和 $x_b$ 的距离，定义为它们的 embedding word vector 之差的 $p$-范数。

公式（11）：对于词 $x_i$，这个损失一方面希望它和同义词们形成的 bound 尽量小(**同义词更紧凑**)，另一方面希望它和随机采样的非同义词保持一定距离(**非同义词别太近**)；如果非同义词已经足够远，就不再继续强行推远。

* $d_{bound}(x_i, S(x_i))$: 词及其同义词集合的 边界大小
* $∥⋅∥_p$: $p$-范数
* $\mathcal{N}(M)$: 从词汇表中随机抽取的 $M$ 个非近义词集合，作为负样本集合。
* $\alpha$: 截断阈值, 一旦距离超过 $\alpha$，这些非近义词就不再被继续推远。
* $min(d(x_i,\tilde{x}_i), \alpha)$: 如果 $d(x_i,\tilde{x}_i)<\alpha$，就取实际距离; $d(x_i,\tilde{x}_i)\ge \alpha$，就只记成 $\alpha$


## 4.3 Overall Training Process

$$\mathcal{L}_{emb} = \frac{1}{|X|} \sum_{x_i \in X} \mathcal{L}_{EIBC}(x_i, S(x_i), \mathcal{N}(M)) \tag{12}$$

$$\mathcal{L}_{model} = (1 - \beta) \cdot \mathcal{L}_{CE} + \beta \cdot \mathcal{L}_{IBP}(\epsilon) \tag{13}$$

公式（12）: 整个句子的 embedding loss = 句子中每个词的 EIBC loss 的平均值

公式（13）:

* $\mathcal{L}_{emb}$: 总词嵌入损失（Final Embedding Loss）。表示整个输入序列在词嵌入微调阶段的总损失。
* $|x|$: 句子中，词的数量
* $\mathcal{L}_{model}$: 模型训练总损失
* $\beta$: 平衡系数（权重超参数）。用于调节 标准准确率损失 和 鲁棒性损失 之间的相对权重。
* $\mathcal{L}_{CE}$: 普通 cross-entropy loss, 衡量模型在正常输入上的分类误差。
* $\mathcal{L}_{IBP}(\epsilon)$: 基于 interval bound propagation 得到的鲁棒损失。衡量模型在给定扰动范围 $\epsilon$ 下的 worst-case 分类损失。





