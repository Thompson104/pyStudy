# -*- coding: utf-8 -*-
"""
程序说明：采用蒙特卡洛方法对样本数据进行扩充，然后进行识别，考察大量样本数据下的识别效率和准确率。
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy import  io as spi
import time as tm
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,train_test_split,validation_curve

from sklearn.preprocessing import StandardScaler  #归一化
from sklearn.decomposition import PCA
def load_Laman():
    '''加载拉曼谱图数据'''
    raw_data = spi.loadmat('../input/XJlaman_4_1.mat')
    x = raw_data['lamanxiangjing']
    y = np.zeros((x.T.shape[0]))
    for i in np.arange(0,30):
        y[i*30:i*30+30,]= i
    return x.T,y

def load_IMS():
    '''加载离子迁移谱数据'''
    raw_data = spi.loadmat('../input/XJlizi_4_1_1.mat')
    x = raw_data['XJlizi_4_1_1']
    y = np.zeros((x.T.shape[0]))
    for i in np.arange(0,30):
        y[i*30:i*30+30,]= i
    return x.T,y
'''
将数据进行扩充：
1）每个维度的均值*0.1*随机数
2）加上原值
'''
def Create_Raman_Data(raw_data_x,raw_data_y,muls=10):
    '''每个数据扩展为mul个数据'''
    if muls == 1:
        return raw_data_x,raw_data_y
    mul = muls
    [m,n] = raw_data_x.shape
    # 构造[10*m，n]矩阵
    random_mean = np.ones((m*mul,n)) # 基于均值的随机波动矩阵
    mul_raw_data = np.ones((m*mul,n))
    mul_raw_data_y = np.ones((m*mul,1))
    # raw_data 扩充mul倍
    for i in range(m):
        mul_raw_data[i*mul:(i+1)*mul,: ] = raw_data_x[i, :]
        mul_raw_data_y[i*mul:(i+1)*mul ] = raw_data_y[i]
    # 各维度平均值
    mean_data = np.mean(raw_data_x, axis=0)
    #
    for i in range(m*mul):
        tp =  mean_data * 0.001 * (np.random.ranf(n) - 0.5)
        random_mean[i, :] = tp + random_mean[i, :]
    return (random_mean + mul_raw_data),mul_raw_data_y

[raw_x_ims,raw_y_ims] = load_IMS()
[raw_x_raman,raw_y_raman] = load_Laman()
# raw_x = np.concatenate((raw_x_raman,raw_x_ims),axis=1)
raw_x = raw_x_raman
raw_y = raw_y_ims

muls = 100
# 取第一个样本的数据进行扩成
[x1,y1] = Create_Raman_Data(raw_x_raman[0:30,:],raw_y_raman[0:30],muls=muls)
# 保存生成的数据
np.savetxt('x1.txt',x1)
np.savetxt('y1.txt',y1)

[x2,y2] = Create_Raman_Data(raw_x_raman[90:120,:],raw_y_raman[90:120],muls=muls)
# 保存生成的数据
np.savetxt('x2.txt',x2)
np.savetxt('y2.txt',y2)


x1 = np.loadtxt('x1.txt')
y1 = np.loadtxt('y1.txt')
x2 = np.loadtxt('x2.txt')
y2 = np.loadtxt('y2.txt')

x = np.concatenate((x1,x2),axis=0)
y = np.concatenate((y1,y2),axis=0)

# 归一化处理
scaler = StandardScaler()
x = scaler.fit_transform(x)

# PCA降维
pca = PCA(n_components=10)
pca.fit(x)
x = pca.transform(x)

# svm 识别
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                                    random_state=0)
# 训练样本的数量
train_num = 2200

print(train_num)
# 构造训练与测试数据集合
x_train = np.concatenate((x1[0:train_num,:],x2[0:train_num,:]),axis=0)
y_train = np.concatenate((y1[0:train_num],y2[0:train_num]),axis=0)

x_test = np.concatenate((x1[train_num:,:],x2[train_num:,:]),axis=0)
y_test = np.concatenate((y1[train_num:],y2[train_num:]),axis=0)
# 记录训练模型耗时
t1 = tm.time()
model = SVC(C=0.001, kernel='rbf')
model.fit(x_train,y_train)
t2 = tm.time()
# 预测测试样本
y_test_1 = model.predict(x_test)
# 计算识别比例
result = np.sum(y_test == y_test_1) / y_test_1.shape[0]

print('训练样本数量:',x_train.shape[0])
print('测试样本数量:',x_test.shape[0])
print('训练模型耗时：%f秒'%((t2-t1)))
print('识别率',result)


'''
results = np.zeros((30,3))
i = 0
for train_num in np.arange(1900,2000,500):
    print(train_num)
    # 构造训练与测试数据集合
    x_train = np.concatenate((x1[0:train_num,:],x2[0:train_num,:]),axis=0)
    y_train = np.concatenate((y1[0:train_num],y2[0:train_num]),axis=0)

    x_test = np.concatenate((x1[train_num:,:],x2[train_num:,:]),axis=0)
    y_test = np.concatenate((y1[train_num:],y2[train_num:]),axis=0)
    # 记录训练模型耗时
    t1 = tm.time()
    model = SVC(C=0.001, kernel='rbf')
    model.fit(x_train,y_train)
    t2 = tm.time()
    # 预测测试样本
    y_test_1 = model.predict(x_test)
    # 计算识别比例
    result = np.sum(y_test == y_test_1) / y_test_1.shape[0]

    print('训练样本数量:',x_train.shape[0])
    print('测试样本数量:',x_test.shape[0])
    print('训练模型耗时：%f秒'%((t2-t1)))
    print('识别率',result)
    results[i:,] = [train_num,result,(t2-t1)]
    i= i+1

plt.plot(results[:,0],results[:,1],'bo')
# plt.plot(results[:,0],results[:,2],'r+')
plt.show()
'''




