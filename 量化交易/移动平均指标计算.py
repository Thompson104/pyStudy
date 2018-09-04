# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:50:39 2017

@author: TIM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rawdata = pd.read_csv('usdjpy_m1.csv')

# 移动平均线
# 分别计算5日、20日、60日的移动平均线
ma_list = [5, 20, 60]
for ma in ma_list:
    rawdata['MA_' + str(ma)] = rawdata['close'].rolling(window=ma).mean()

# 指数平滑移动平均线EMA
# 分别计算5日、20日、60日的移动平均线
    ma_list = [5, 20, 60]
for ma in ma_list:
    rawdata['EMA_' + str(ma)] = rawdata['close'].ewm(span=ma).mean()
    
# =====================================================
# numpy计算移动平均值
N=5
n=np.ones(N)
weights=n/N
# 利用卷积函数
sma=np.convolve(weights,rawdata['close'].values)[N-1:-N+1]

