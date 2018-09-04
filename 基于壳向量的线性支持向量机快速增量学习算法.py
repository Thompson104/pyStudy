# -*- coding: utf-8 -*-
"""
基于壳向量的线性支持向量机快速增量学习算法
"""
import numpy as np
#import scipy.io as spi
from sklearn import svm 
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
'''
This module was deprecated in version 0.18 in favor of the model_selection module 
into which all the refactored classes and functions are moved. Also note that 
the interface of the new CV iterators are different from that of this module. 
This module will be removed in 0.20.
'''
#from sklearn import cross_validation as cv 

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#======================================================
# 加载原始数据集
raw_data = np.loadtxt('./input/bupa.data',delimiter=',')
x = raw_data[:,:-1]
y = raw_data[:,-1]

scaler = StandardScaler()
x = scaler.fit_transform(x)

train_x,test_x,train_y,test_y = train_test_split(x,y,train_size=105,random_state=19)

#=======================================================
# 训练模型
clf1 = svm.SVC()
model1 = clf1.fit(train_x,train_y)
result_y = clf1.predict(test_x)
result = result_y - test_y
print('svm.SVC正确率：%f'%(np.sum(result == 0)/result.shape[0]))
print('类别：%d 的支持向量数量为%d'%(1,clf1.n_support_[0]))
print('类别：%d 的支持向量数量为%d'%(2,clf1.n_support_[1]))

# 1、分别针对数据集A+和A-求壳向量集A+HV和A-HV,令AHV= A+HV∪ A-HV
# 不同类别的数据分开
A_plus = train_x[train_y == 1]
A_minus = train_x[train_y == 2]
# 求不同类别数据集合的壳向量
A_hv_plus = ConvexHull(A_plus)
A_hv_minus = ConvexHull(A_minus)
# 将壳向量合并
A_hv = np.concatenate((A_plus[A_hv_plus.vertices,:], A_minus[A_hv_minus.vertices,:] ),axis=0)
A_hv_label = np.concatenate(((np.zeros(A_hv_plus.vertices.shape[0]) +1 ),
                             (np.zeros(A_hv_minus.vertices.shape[0]) +2  )  ),axis=0)

# 2、将壳向量集AHV作为新的训练样本集,运行SVM算法得到支持向量集ASV,并由此构造最优分类决策函数
clf2 = svm.SVC(C=8.25,gamma=0.0316)
model2 = clf2.fit(A_hv,A_hv_label)
result_y = clf2.predict(test_x)
result = result_y - test_y
print('壳向量集作为新的训练样本集,正确率：%f'%(np.sum(result == 0)/result.shape[0]))
print('类别：%d 的支持向量数量为%d'%(1,clf2.n_support_[0]))
print('类别：%d 的支持向量数量为%d'%(2,clf2.n_support_[1]))
print('类别：%d 的壳向量数量为%d'%(2,A_hv.shape[0]))

# 3、开始增量学习
nums = 40
for i in np.arange(0,5):
    print(i*nums,(i+1)*nums)
    A = np.concatenate((test_x[i*nums:(i+1)*nums,:],A_hv),axis=0) # 令A= B∪ AHV作为新训练样本集
    A_label = np.concatenate([test_y[i*nums:(i+1)*nums],A_hv_label])
    A_plus = A[A_label == 1]
    A_minus = A[A_label == 2]
    A_hv_plus = ConvexHull(A_plus)
    A_hv_minus = ConvexHull(A_minus)
    # 将壳向量合并
    A_hv = np.concatenate((A_plus[A_hv_plus.vertices, :], A_minus[A_hv_minus.vertices, :]), axis=0)
    A_hv_label = np.concatenate(((np.zeros(A_hv_plus.vertices.shape[0]) + 1),
                             (np.zeros(A_hv_minus.vertices.shape[0]) + 2)), axis=0)
    clf = svm.SVC()
    model = clf.fit(A_hv,A_hv_label)
    # result_y = clf.predict(test_x[5*nums:(5+1)*nums,:])
    # result = result_y - test_y[5*nums:(5+1)*nums]
    # print('正确率：%f'%(np.sum(result == 0)/result.shape[0]))
    # print('类别：%d 的支持向量数量为%d'%(1,clf2.n_support_[0]))
    # print('类别：%d 的壳向量数量为%d' % (1, A_hv_plus.vertices.shape[0]))
    # print('类别：%d 的支持向量数量为%d'%(2,clf2.n_support_[1]))
    # print('类别：%d 的壳向量数量为%d' % (1, A_hv_minus.vertices.shape[0]))

    result_y = clf.predict(x)
    result = result_y - y
    print('正确率：%f'%(np.sum(result == 0)/result.shape[0]))
    print('类别：%d 的支持向量数量为%d'%(1,clf2.n_support_[0]))
    print('类别：%d 的壳向量数量为%d' % (1, A_hv_plus.vertices.shape[0]))
    print('类别：%d 的支持向量数量为%d'%(2,clf2.n_support_[1]))
    print('类别：%d 的壳向量数量为%d' % (1, A_hv_minus.vertices.shape[0]))


'''

# 取出不同类别的支持向量
sv1 = clf1.support_vectors_[0:81,:]
sv2 = clf1.support_vectors_[81:,:]

# 取出所有顶点
hv1 = ConvexHull(sv1)
print('类1点的个数为：%d'% (hv1.npoints))
print('类1顶点的个数为：%d'% (hv1.vertices.shape[0]))
hv1 = hv1.points[hv1.vertices,:]

hv2 = ConvexHull(sv2)
print('类2点的个数为：%d'% (hv2.npoints))
print('类2顶点的个数为：%d'% (hv2.vertices.shape[0]))
hv2 = hv2.points[hv2.vertices,:]
#=======================================================
# 开始增量学习
#for c in test_x:
#    print(c)

    



#=======================================================
'''