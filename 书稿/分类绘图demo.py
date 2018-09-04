all# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def plot_classifier(classifier,X,y):
    x_min,x_max = np.min(X[:,0]) - 1.0, np.max(X[:,0]) + 1.0
    y_min,y_max = np.min(X[:,1]) - 1.0, np.max(X[:,1]) + 1.0
    step_size = 0.01
    x_values,y_values = np.meshgrid(np.arange(x_min,x_max,step_size),
                                    np.arange(y_min,y_max,step_size))
    #np.c_将切片对象沿第二个轴（按列）转换为连接。
    mesh_out = classifier.predict(np.c_[x_values.ravel(),y_values.ravel()])
    mesh_out =mesh_out.reshape(x_values.shape)
    plt.figure()
    # 绘制四边形网格,功能类似于 pcolor(),
    # cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。
    # http://blog.csdn.net/haoji007/article/details/52063168
    plt.pcolormesh(x_values,y_values,mesh_out,cmap=plt.cm.gray)
    plt.scatter(X[:,0],X[:,1],c=y,s=80,edgecolors='black',linewidths=1,cmap=plt.cm.Paired)
    plt.xlim(x_values.min(),x_values.max())
    plt.ylim(y_values.min(),y_values.max())
    
    plt.xticks(np.arange(int(min(X[:,0])-1),
                         int(max(X[:,0])+1),1.0
                         )
               )
    plt.yticks(np.arange(int(min(X[:,1])-1),
                         int(max(X[:,1])+1),1.0
                         )
               )
    plt.show()
    
    

X = np.array([[4,7],
              [3.5,8],
              [3.1,6.2],
              [0.5,1],
              [1,2],
              [1.2,1.9],
              [6,2],
              [5.7,1.5],
              [5.4,2.2]])
y = np.array([0,0,0,1,1,1,2,2,2])

classifier = linear_model.LogisticRegression(solver='liblinear',C=100)

classifier.fit(X,y)
plot_classifier(classifier,X,y)

