# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 07:34:17 2017

@author: TIM
"""

import numpy as np
import scipy as sp
import random
import scipy.io as sps
import pandas as pd
from sklearn.neural_network import multilayer_perceptron
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing

# 1\ 加载数据
rawdata = sps.loadmat('.\\input\spectra.mat')

x = rawdata['NIR']
y = rawdata['octane']

# 观察原始数据
#plt.plot(x.T)
# 从0~60中随机选取40个数

train_index = random.sample(range(0,60),60)
'''
x_train = x[train_index[0:40]]
y_train = y[train_index[0:40]]

x_test = x[train_index[40:]]
y_test = y[train_index[40:]]
'''
# 数据标准化（归一化）处理是数据挖掘的一项基础工作

'''
# min-max标准化（Min-Max Normalization）
min_max_scaler  = preprocessing.MinMaxScaler()
min_max_scaler.fit(x)
x_scaled = min_max_scaler.transform(x)

x_min = np.min(x,axis=0)
x_max = np.max(x,axis=0)
ranges = x_max - x_min
x_scaled = (x -x_min) / ranges
'''

'''
standardScaler = preprocessing.StandardScaler()
standardScaler.fit(x)
#x_scaled = standardScaler.transform(x)
'''

# z-score规范化
x_scaled = preprocessing.scale(x,axis=1) # good


'''
plt.plot(list(range(0,401)),x[0,:])
plt.plot(list(range(0,401)),x_scaled[0,:])
plt.grid()
plt.show()
'''
# 构造测试数据集、训练数据集
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,random_state = 1,train_size=0.8)


# 2\ 建立神经网络模型

model = multilayer_perceptron.MLPRegressor()
model.fit(x_train,y_train)

# 3、使用模型进行预测
# 将(12,)转换为(12,1)维度
result = np.array( model.predict(x_test),ndmin=2).T  #方法1
result = model.predict(x_test).reshape(-1,1)        #方法2

# 4、分析结果
# 比较结果，计算误差
#result1 =np.array([result, y_test.reshape(12), (( y_test.reshape(12)-result)/ result) ]).T

loss = (y_test - result)/ y_test
result1 =np.concatenate((result,y_test,loss),axis=1)
#scores = model.score(x_test,y_test.reshape(20))
scores = model.score(x_test[0].reshape(1,-1),y_test[0].reshape(1,-1))

# 5、作图分析

'''
plt.plot(list(range(0,12)),result1[:,0])
plt.plot(list(range(0,12)),result1[:,1])
plt.grid()
plt.show()
'''



