# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:04:05 2018

@author: Tim
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%%
def data_helper():
    data = ds.load_iris()
    train_x,test_x,train_y,test_y = train_test_split(data.data,data.target,
                                                     stratify=data.target,
                                                     test_size=0.3)
    return train_x,test_x,train_y,test_y
def plot_helper(X,y):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    colors='rgb'
    markers='o*s'
    for target,color,marker in zip([0,1,2],colors,markers):
        x = X[y==target,:]
        ax.scatter(x[:,0],x[:,1],x[:,2],color=color,marker=marker,
                   label = '标签 %s'%target)
        ax.set_xlabel('x轴',fontproperties=myfont)
        ax.set_ylabel('y轴',fontproperties=myfont)
        ax.set_zlabel('z轴',fontproperties=myfont)
        ax.legend(loc='best',prop=myfont)
    fig.suptitle('Iris 数据',fontproperties=myfont)
    return

train_x,test_x,train_y,test_y = data_helper()


#plt.scatter(train_x[:,0],train_x[:,3],c=train_y)

lda = LinearDiscriminantAnalysis()
lda.fit(train_x,train_y)
#print(lda.Covariance)
print(lda.score(test_x,test_y))

#plot_helper(x,train_y)
plt.scatter(x[:,0],x[:,1])