# -*- coding: utf-8 -*-
"""
标签传播算法
相关网址：http://blog.csdn.net/zouxy09/article/details/49105265
@author: TIM
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import label_propagation
from sklearn.datasets import make_circles

n_samples = 200
# 生产两个圆环数据，用于分类算法
# x[0]，x[1]为两个坐标，前面一半数据为一个圆环，后一半数据为另一个圆环
# noise=0.003以上，分类效果开始变差
x,y = make_circles(n_samples=n_samples,shuffle=False,noise=0.003)
outer ,inner = 0,1
labels = - np.ones(n_samples)
# 第一个数据的标签为outer，
labels[0] = outer
# 最后一个数据的标签为inner
labels[-1] = inner


label_spread = label_propagation.LabelSpreading(kernel='knn',alpha=1.0)
label_spread.fit(x,labels)

output_labels = label_spread.transduction_
plt.figure(figsize=(8.5,4))
plt.subplot(1,2,1)
plt.scatter(x[labels == outer,0],x[labels == outer,1],color='navy',
            marker='s',lw=0,label='outer labeled',s=10)
plt.scatter(x[labels == inner,0],x[labels == inner,1],color='c',
            marker = 's',lw=0,label='inner labeled',s= 10)
plt.scatter(x[labels == -1,0],x[labels == -1,1],color='darkorange',
            marker='.',label='unlabeled')
plt.legend(scatterpoints=1,shadow=False,loc='upper right')
plt.title('Raw data (2 classes=outer and inner)')

plt.subplot(1,2,2)
output_label_array = np.asarray(output_labels)
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]

plt.scatter(x[outer_numbers,0],x[outer_numbers,1],color='navy',
            marker='s',lw=0,s=10,label='outer learned')

plt.scatter(x[inner_numbers,0],x[inner_numbers,1],color='c',
            marker='s',lw=0,s=10,label='inner learned')
plt.legend(scatterpoints=1,shadow=False,loc='upper right')
plt.title('Labels learned with Label Spreading (knn)')

plt.subplots_adjust(left=0.07,bottom=0.07,right=0.93,top=0.92)
plt.show()




