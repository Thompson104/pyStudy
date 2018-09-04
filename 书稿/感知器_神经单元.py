# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:51:52 2018

@author: Tim
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%%
class Perceptron(object):
    '''
    eta:学习率
    n_iter：权重向量的训练次数
    w_:神经元权重
    '''
    def __init__(self,eta=0.01,n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = []
        return
    def fit(self,X,y):
        '''
        X:shape[n_samples,n_features]
        y:shape[n_samples]
        '''
        # 初始化权重向量为0，权重向量w0为阈值
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ =[]
        # 开始训练
        for _ in range(self.n_iter):
            errors = 0
            # 利用每个样本进行训练
            for xi,target in zip(X,y):
                # 计算权重修正值
                # data_w = nu * (y - y')
                data_w = self.eta * (target - self.predict(xi))     
                # 更新权重
                self.w_[1:] = self.w_[1:] + data_w * xi 
                # 更新阈值
                self.w_[0]  = self.w_[0] + data_w * 1
                # 记录错误的次数
                errors += int(data_w != 0.0)
                self.errors_.append(errors)    
                pass
            pass
        return
    # 分类预期函数
    def predict(self,X):
        y = np.where(self.net_input(X) >=0,1,-1)
        return y
        
    # 根据神经元输入的数据，进行计算
    def net_input(self,X):
        # z = w0 * 1 + w1 * x1 +...+ Wn *Xn
        result = np.dot(X,
                        self.w_[1:])  + self.w_[0]
        return result
        
    pass

def plot_decision_regions(X,y,classifier,resolution=0.02):
    '''
    绘制感知器线性分类器的图形
    '''
    markers = ('s','x','o','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min,x1_max = X[:,0].min() -1 , X[:,0].max()  
    x2_min,x2_max = X[:,1].min() -1, X[:,1].max()   
    
    ## 坐标平面的网格
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))

    temp =  np.array([ xx1.ravel(),xx2.ravel() ]).T
    z = classifier.predict( temp )
    print(xx1.ravel())
    print(xx2.ravel())
    print(z)
    
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.5,c=cmap(idx),marker=markers[idx],label=cl)
    return

if __name__ == '__main__':
    #%% 加载数据
    data_file_name = r'..\input\iris.data'
    df = pd.read_csv(data_file_name,header=None)
    X = df.values[:,[0,2]]
    y =df.values[:,4]
    print(np.unique(y)) # 观察y
    y = np.where( y == 'Iris-setosa',1,-1 )
    print(y.dtype)
    #%%
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=y,marker='o')
    plt.xlabel('花瓣长度',fontproperties=myfont)
    plt.ylabel('花茎长度',fontproperties=myfont)
    plt.legend(loc='best',title='图例',prop=myfont)
    plt.show()
    
    plt.figure()
    pnn = Perceptron()
    pnn.fit(X,y)
    plot_decision_regions(X,y,pnn,resolution=0.02)
    pass





