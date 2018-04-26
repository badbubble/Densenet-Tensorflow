# Densenet Tensorflow 实现
---
因为最近很多比赛要用,所以看了Densenet的论文(我看的第一篇论文), 打算用tensorflow实现一下(抄别人的), 加深一下自己的理解.
顺便记个笔记

## 原理知识点
---
### [global average pooling](https://blog.csdn.net/yimingsilence/article/details/79227668)
相对于average pooling 它是对所的channel进行的, 如果有1000个类， 最后一个卷积层输出1000个channel, 对每个channel进行GAP, 剔除了全
连接层的暗箱, 赋予每个channel实际意义

### [Batch Normalization ](https://blog.csdn.net/hjimce/article/details/50866313)

&nbsp;(抄自博客)我们知道在神经网络训练开始前，都要对输入数据做一个归一化处理，那么具体为什么需要归一化呢？归一化后有什么好处呢？原因在于神经网络学习过程本质就是为了学习数据分布，一旦训练数据与测试数据的分布不同，那么网络的泛化能力也大大降低；另外一方面，一旦每批训练数据的分布各不相同(batch 梯度下降)，那么网络就要在每次迭代都去学习适应不同的分布，这样将会大大降低网络的训练速度，这也正是为什么我们需要对数据都要做一个归一化预处理的原因
在训练过程中分布可能会发生改变，Batch Normalization就是要解决在训练过程中，中间层数据分布发生改变的情况

&nbsp;算法本质原理就是这样：在网络的每一层输入的时候，又插入了一个归一化层，也就是先做一个归一化处理，然后再进入网络的下一层。不过文献归一化层，可不像我们想象的那么简单，它是一个可学习、有参数的网络层。

## 代码知识点
---

### [arg_scope](https://blog.csdn.net/u012436149/article/details/72852849)


```Python
tf.contrib.framework.arg_scope(list_ops_or_scope, **kwargs)  # 为给定的 list_ops_or_scope 存储默认的参数
```


## 未解决的问题
* [ ] 整理代码结构
* [ ] Batch Normalization的具体原理以及代码部分(看一下概率论)
