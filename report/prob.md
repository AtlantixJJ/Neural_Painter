问题描述：

1. 使用GAN进行的，半监督下学习出face segmentation的方法。应该能够进行比较得到更加准确的结果。

2. 实现方法是一个现有的NIM GAN网络，中间的feature层出来一个到segmentation的分类层，这个分类层还可以添加到输入层。即增加seg_feat, feat_seg层。

3. 以后可以考虑减少seg_feat的深度，将其和noise合并起来。