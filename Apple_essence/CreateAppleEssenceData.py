# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy import  io as spi
from matplotlib import pyplot as plt

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

muls = 2
# 取第一个样本的数据进行扩成
[x1,y1] = Create_Raman_Data(raw_x_raman[0:30,:],raw_y_raman[0:30],muls=muls)
# 保存生成的数据
np.savetxt('x1.txt',x1)
np.savetxt('y1.txt',y1)

[x2,y2] = Create_Raman_Data(raw_x_raman[90:120,:],raw_y_raman[90:120],muls=muls)
# 保存生成的数据
np.savetxt('x2.txt',x2)
np.savetxt('y2.txt',y2)