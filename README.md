# 工业界GAUC计算总结

## 背景

> auc在传统的机器学习二分类中还是很能打的，但是有一种场景，虽然是分类模型，但是却不适用auc，即广告推荐领域。
推荐领域使用的CTR（点击率）来作为最终的商用指标，但是在训练推荐模型时，却不用这个指标，用的是GAUC。
原因是推荐模型目前比较成熟的模式是训练分类模型，这个分类模型的任务是预测用户是否会点击给定的商品，因此，推荐系统的核心，仍然是一个二分类问题，但是是更细力度的二分类。

参考：https://zhuanlan.zhihu.com/p/84350940

## GAUC

> GAUC（group auc）是计算每个用户的auc，然后加权平均，最后得到group auc，这样就能减少不同用户间的排序结果不太好比较这一影响。

> GAUC具体公式如下：

$$ GAUC = \sum_{i=1}^n w_{i} \cdot AUC_{i} / \sum_{i=1}^n w_{i} $$

$$ w_{i} 是每个用户的权重 $$

可以设为每个用户的impression或click的次数，在计算每个用户的auc时，会过滤掉单个用户全是正样本或负样本的情况

### AUC计算

#### 二分类指标的AUC计算

> 对于二分类指标，AUC可直接调用sklearn的roc_auc_score计算

#### 非二分类指标的AUC计算

> 短视频领域往往涉及到非分类指标，如播放时长，模型预测播放时长和视频实际播放时长都是连续值，无法直接调用sklearn的roc_auc_score计算AUC，因此对于该类指标往往采用逆序对数量来量化两个排序列表的不一致程度，从而定量表征模型推荐效果。

#### 肯德尔等级相关系数（Kendall Tau）

> 统计学里量化两个排序列表不一致程度的指标是肯德尔等级相关系数（Kendall Tau），肯德尔等级相关系数有三种方式，下面介绍最简单的一种，计算公式如下：

$$ Kendall \ Tau = \frac { N_{c} - N_{d} }{ N_{0}} $$

$$ N_{c}是两个列表中相对顺序保持一致的元素对数量 $$

$$ N_{d}是两个列表中相对顺序不一致的元素对数量 $$

$$ N_{0} = \frac {n \cdot (n-1)}{2}是总的元素对数量，n是元素列表的个数 $$

$$ Kendall \ Tau \in [-1, 1] $$

参考：

https://zhuanlan.zhihu.com/p/63279107

https://blog.csdn.net/qq_39885465/article/details/105539407

> 由于肯德尔等级相关系数范围在[-1, 1]之间，而分类问题的AUC都是在[0, 1]之间，因此参考肯德尔等级相关系进行改进，得到量化两个排序列表不一致程度的指标，如下：

$$ AUC(非二分类) = \frac { N_{c}}{ N_{0}}  \in [0, 1] $$


### 代码

> 普通AUC计算直接调用sklearn的roc_auc_score，非二分类的AUC计算涉及到两个列表中相对顺序保持一致的元素对数量，即列表里逆序对的数量统计，可采用归并排序或者二分查找的方法，具体参考：https://github.com/ShaoQiBNU/GAUC/blob/main/gauc.ipynb

> 工业界用python的dataframe计算效率较低，常采用pyspark进行提速计算，具体参考：https://github.com/ShaoQiBNU/GAUC/blob/main/gauc_pyspark.py

