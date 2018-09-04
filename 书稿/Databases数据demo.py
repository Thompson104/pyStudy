# -*- coding: utf-8 -*-
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=12)

fig = plt.figure(figsize=(10, 8))
# 聚类数据生成
# 根据用户指定的特征数量、中心点数量、范围等来生成具有各向同性的高斯分布数据
data,target = make_blobs(n_samples=100,
                         n_features=2,
                         centers=3,
                         cluster_std=[1.0,3.0,2.0])
ax = fig.add_subplot(221)
plt.title("make_blobs生成的聚类数据", fontproperties=myfont)
plt.scatter(data[:,0],data[:,1],c=target)

# 回归数据生成
# X为样本特征，y为样本输出， coef为回归系数，共500个样本，每个样本1个特征
X, y, coef =make_regression(n_samples=500, n_features=1, noise=5, coef=True)
ax = fig.add_subplot(222)
plt.title("make_regression生成的回归数据", fontproperties=myfont)
plt.scatter(X, y,  color='black')
plt.plot(X, X*coef, color='blue',linewidth=3)

# 分类数据生成
# 单标签分类数据生成
# X为样本特征，y为样本类别输出，400个样本，每个样本2个特征，输出有3个类别
'''
参数说明：
n_features :特征个数= n_informative（） + n_redundant + n_repeated
n_informative：多信息特征的个数，即不相关的特征个数
n_redundant：冗余信息，informative特征的随机线性组合
n_repeated ：重复信息，随机提取n_informative和n_redundant 特征
n_classes：分类类别
n_clusters_per_class ：某一个类别是由几个cluster构成的
'''
X, y = make_classification(n_samples=400, 
                           n_features=5, n_redundant=0,
                           n_informative = 4,
                           n_clusters_per_class=2, n_classes=3)
ax = fig.add_subplot(223)
plt.title("make_classification生成的单标签分类数据", fontproperties=myfont)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

# 分组多维正态分布的数据生成
# 生成2维正态分布，生成的数据按分位数分成3组，1000个样本,
# 2个样本特征均值为1和2，协方差系数为2
'''
参数说明：
n_samples：生成样本数
n_features：正态分布的维数
mean：数据的特征均值
cov：数据协方差的系数
n_classes：数据在正态分布中按分位数分配的组数
'''
X, y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, mean=[1,2],cov=2)
ax = fig.add_subplot(224)
plt.title("make_gaussian_quantiles生成的分组多维正态分布的数据", fontproperties=myfont)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

plt.show()


