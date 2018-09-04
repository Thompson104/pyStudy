# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 22:19:40 2018

@author: TIM
"""

import numpy as np

'''
数据生成函数
'''
def genData(numPoints, # 数据的个数
            bias,       # 偏差,与节距
            variance):
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=numPoints)
    x[:,0] = 1
    for i in np.arange(0,numPoints):
        x[i,1] = i
        y[i] = (i + bias ) + np.random.uniform(0,1) * variance
    return x,y
# ============================================================
'''
梯度下降算法
有问题???/
'''
def gradientDescent(x,y,theta,alpha,m,numIteration):
    xTrans = x.transpose()
    for i in np.arange(0,numIteration):
        hypothesis = np.dot(x,theta)
        loss = hypothesis - y
        #print(loss)
        #
        cost = np.sum(loss ** 2) / ( 2 * m )
        print('Iteration %d  / Cost: %f' % (i,cost))
        # 每个样本求平均梯度
        gradient = np.dot(xTrans,loss) / m
        theta = theta - alpha * gradient
    return theta
# ============================================================
if __name__ == '__main__':
    x,y = genData(100,25,10)
    m,n = np.shape(x)
    n_y = np.shape(y)
    
    numIteration = 10
    alpha = 0.005
    theta = np.ones(n)
    theta = gradientDescent(x,y,theta,alpha,m,numIteration)
    print(theta)
    