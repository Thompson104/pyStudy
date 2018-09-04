# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:10:07 2017

@author: TIM
"""
# %cd "z:\"
import sys
import numpy as np
# 读取文件，价格及成交量
c,v = np.loadtxt('data.csv',delimiter=',',
                 usecols=(6,7),unpack=True)

# 计算成交量加权平均价格
vwap = np.average(c,weights=v)

print('加权平均 VWap = ',vwap)

print('算术平均 mean =' ,np.mean(c))

# 时间加权平均价格
t = np.arange(len(c)) # 根据交易的先后，构造一个数字序列
print('twap =',np.average(c,weights=t))

# 寻找最大值和最小值
# 最高价和最低价
h,l = np.loadtxt('data.csv',delimiter=',',usecols=(4,5),unpack=True)

print('highest price =',h.max())

print('最高成交价的波动范围 = ',h.ptp())
print('最低成交价的波动范围 = ',l.ptp())

# 中位数
print('中位数 = ',np.median(h))
# 方差
print('方差 = ',np.var(h))
print('方差 = ', np.mean( (h-np.mean(h))**2))

# 股票收益率
# np.diff out[n] = a[n+1] - a[n]
returns = np.diff(c) / c[:-1]

# 对数收益率
if (c <= 0).sum() == 0:# 确保没有0或小于零，否则对数报错
    logreturns = np.diff(np.log(c))

# 收益大于0的位置
posret_indices = np.where(returns > 0)

# 波动率
# 年度波动率
annual_valatility = np.std( logreturns ) / np.mean( logreturns )
annual_valatility = annual_valatility / np.sqrt(1. / 252)

# 月度波动率
monthly_valatility = annual_valatility * np.sqrt(1/12)

from datetime import datetime
# 日期分析
def datestr2num(s):
    #s = str(s,encoding='utf-8')
    s = bytes.decode(s)
    return datetime.strptime(s, "%d-%m-%Y").date().weekday()

dates, close=np.loadtxt('data.csv', delimiter=',', usecols=(1,6), 
                         converters={1: datestr2num}, unpack=True)

averages = np.zeros(5)

for i in range(5):
    indices = np.where(dates == i)
    prices = np.take(close,indices)
    avg = np.mean(prices)
    print('Day ',i,' prices',prices,'Average',avg)
    averages[i] = avg




