问题描述：

1. 使用GAN进行的，半监督下学习出face segmentation的方法。应该能够进行比较得到更加准确的结果。

2. 实现方法是一个现有的NIM GAN网络，中间的feature层出来一个到segmentation的分类层，这个分类层还可以添加到输入层。即增加seg_feat, feat_seg层。

3. 以后可以考虑减少seg_feat的深度，将其和noise合并起来。

expr 1

生成器可以使用BN，conditional BN（投影到channel维度，虽然我觉得这个参数化过量了）；
但是判别器完全不能使用任何BN，尝试了conditional BN和普通BN，他们在gamma beta参数不共享时都不行，推测是这两个参数帮助判别器判断了真假（比如假的gamma 和 beta直接将tensor变成0，方便后面分类）。在gamma beta不共享时，只有default BN勉强能够训练出结果，但是效果和收敛速度均慢于没有BN。这个和其他所有CV领域中的结果矛盾，判别器的结构已经很复杂了，BN理论上是用来加速训练的。此处不能加速训练，唯一的可能就是数据分布假设不成立了：生成器的数据分布是一直在变的。
但是将BN中的moving variable针对real fake两个数据类型使用了不共享的变量，那么对于这时BN的数据分布就是确定的了。至少对于real来说是确定的。事实上，我观察到判别器的loss降低的似乎快了一些，标志判别器的能力得到了增强。然而生成器的结果正好相反，所有BN都不能成功训练出生成器，反倒是判别器的loss几乎变成了0.
这就是说判别器给生成器的导数没有什么指导意义。可能是判别器没有从真实数据中学到好的知识（这个可能性比较小，也不会是决定性的因素），可能是判别器给生成器的导数不好（由于BN层的出现，输入的扰动在输出的复杂度直线增加之类的）