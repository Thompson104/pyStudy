# -*- coding: utf-8 -*-
'''
LDA算法
'''
import numpy as np
import sklearn.datasets as ds
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

#%% 
#X,y = ds.make_classification(n_samples=999,
#                             n_features=3,
#                             n_informative=2,
#                             n_redundant=0,
#                             n_classes=3,
#                             n_clusters_per_class=1,
#                             random_state=0)
X, y = ds.make_classification(n_samples=998, 
                              n_features=3,
                              n_redundant=0, 
                              n_classes=2, 
                              n_informative=2,
                              n_clusters_per_class=1,
                              class_sep =.5, 
                              random_state =10)

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o',c=y)
plt.show()
#%%
'''
可以看出降维后样本特征和类别信息之间的关系得以保留。
一般来说，如果我们的数据是有类别标签的，那么优先选择LDA去尝试降维；
当然也可以使用PCA做很小幅度的降维去消去噪声，然后再使用LDA降维。如果没有类别标签，
那么肯定PCA是最先考虑的一个选择了
'''
lda = LDA(n_components=1)
lda.fit(X,y)
X_new = lda.transform(X)
#plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.plot(X_new[:,0],'r.')
plt.show()
#%%
'''
由于PCA没有利用类别信息，我们可以看到降维后，样本特征和类别的信息关联几乎完全丢失
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
# PCA找到的两个主成分方差比和方差
print(pca.explained_variance_ratio_)
print( pca.explained_variance_)

X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o',c=y)
plt.show()