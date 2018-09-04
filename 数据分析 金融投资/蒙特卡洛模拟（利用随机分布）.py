# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:49:04 2017

@author: TIM
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
####模拟####
# 随机变量

S0 = 100  # initial value
r = 0.05  # constant short rate
sigma = 0.25  # constant volatility
T = 2.0  # in years
I = 10000  # number of random draws
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T 
             + sigma * np.sqrt(T) * npr.standard_normal(I))

plt.hist(ST1, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

# 对数正态分布
ST2 = S0 * npr.lognormal((r - 0.5 * sigma ** 2) * T,
                        sigma * np.sqrt(T), size=I)

plt.hist(ST2, bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

# 通过统计分析进行比较
import scipy.stats as scs

def print_statistics(a1, a2):
    ''' Prints selected statistics.
    
    Parameters
    ==========
    a1, a2 : ndarray objects
        results object from simulation
    '''
    sta1 = scs.describe(a1)
    sta2 = scs.describe(a2)
    print( "%14s %14s %14s" %         ('statistic', 'data set 1', 'data set 2'))
    print( 45 * "-")
    print( "%14s %14.3f %14.3f" % ('size', sta1[0], sta2[0]))
    print( "%14s %14.3f %14.3f" % ('min', sta1[1][0], sta2[1][0]))
    print( "%14s %14.3f %14.3f" % ('max', sta1[1][1], sta2[1][1]))
    print( "%14s %14.3f %14.3f" % ('mean', sta1[2], sta2[2]))
    print( "%14s %14.3f %14.3f" % ('std', np.sqrt(sta1[3]), np.sqrt(sta2[3])))
    print( "%14s %14.3f %14.3f" % ('skew', sta1[4], sta2[4]))
    print( "%14s %14.3f %14.3f" % ('kurtosis', sta1[5], sta2[5]))

print_statistics(ST1, ST2)

####随机过程###
#布朗运动
I = 10000
M = 50
T = 2.0  # in years
dt = T / M
I = 10000  # number of random draws
S = np.zeros((M + 1, I))
S[0] = S0
for t in range(1, M + 1):
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt 
            + sigma * np.sqrt(dt) * npr.standard_normal(I))
            
plt.hist(S[-1], bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

print_statistics(S[-1], ST2)

plt.plot(S[:, :10], lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.grid(True)



##真实数据
import pandas as pd
import pandas_datareader.data as web


symbols = ['^GDAXI', '^GSPC', 'YHOO', 'MSFT']
symbols = ['YHOO']
data = pd.DataFrame()
for sym in symbols:
    data[sym] = web.DataReader(sym, data_source='yahoo',
                            start='1/1/2013')['Adj Close']
data = data.dropna()

data.info()