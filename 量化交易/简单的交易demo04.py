# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 23:43:52 2018

@author: smart
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import tushare as ts
from statsmodels.tsa.stattools import adfuller

start = '2015-01-01'
end   = '2017-11-01'

stock1 = '601618'
stock2 = '600026'

a1 = ts.get_hist_data(stock1,start,end)
a1= a1['close']
a2 = ts.get_hist_data(stock2,start,end)
a2= a2['close']

print(a1.shape,a2.shape) # 不一定相等

# 处理a1与a2中的缺失值，
stock = pd.DataFrame()
stock['a1'] = a1
stock['a2'] = a2

stock = stock.dropna()
a1 = stock['a1']
a2 = stock['a2']

plt.scatter(a1.values,a2.values)
plt.xlabel(stock1)
plt.ylabel(stock2)
print(np.corrcoef(a1,a2))

# 价差
a3 = a1 - a2
a3.plot(figsize=(10,5))

# 检查价差的平稳性
# adf 单位根验证，观察p-value
adf_test = adfuller(a3)
result = pd.Series(adf_test[0:4],index=['Test Statistic','p-value','usedlag','nobs'])
for key,value in adf_test[4].items():
    result['Critical Values: %s'%key] = value
print(result)

#
    