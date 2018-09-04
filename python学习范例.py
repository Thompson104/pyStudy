# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:10:11 2017

@author: TIM
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm
import seaborn

import scipy as sp
from scipy import stats
from scipy.optimize import leastsq
import scipy.optimize as opt
from scipy import stats
from scipy.stats import norm,poisson
from scipy.interpolate import BarycentricInterpolator
from scipy.interpolate import CubicSpline
from sklearn.model_selection import train_test_split

import  math
import time

if __name__ == '__main__':
    np.set_printoptions(linewidth=200,suppress=True)
    '''
    单行与单列的相加，结果为一个矩阵.
    广播机制的作用
    '''
#    a = np.arange(0,60,10).reshape((-1,1)) + np.arange(6)
#    print(a)
    
    # 从[0,100),步长为10.5
    a = np.arange(0,100,10.5) 
    # 从[0,100),产生500个数字
    b = np.linspace(0,100,500)
    # 从10^1 到了10^4,有4个数的等比序列,默认包含终点
    c= np.logspace(1,10,4,endpoint=True,base=10)
    
    s = 'abcd'
    g = np.fromstring(s,dtype=np.int8)
    
    t = 10000
    a = np.zeros(10000)
    for i in range(t):
        a += np.random.uniform(-5,5,10000)
    a = a/ t
#    b=plt.hist(a,bins= 61,color='g',alpha=.5,normed=True)
#    plt.grid(True)
#    plt.show()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    from numpy.random import randn
    plt.plot(randn(50).cumsum(), 'k--')
    _ = ax1.hist(randn(100), bins=20, color='k', alpha=0.3)
    ax2.scatter(np.arange(30), np.arange(30) + 3 * randn(30))
    plt.close('all')
    
    fig, axes = plt.subplots(2, 3)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=None)
    
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            axes[i, j].hist(randn(500), bins=50, color='k', alpha=0.5)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.close('all')
    # 线条格式    
    plt.plot(randn(30).cumsum(), 'ko--')
    data = randn(30).cumsum()
    plt.plot(data, 'k--', label='Default')
    plt.plot(data, 'k-', drawstyle='steps-post', label='steps')
    plt.legend(loc='best')
    # Setting the title, axis labels, ticks, and ticklabels
    fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
    ax.plot(randn(1000).cumsum())
    
    ticks = ax.set_xticks([0, 250, 500, 750, 1000])
    labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],
                                rotation=45, fontsize='small')
    ax.set_title('some random lines')
    ax.set_xlabel('Stages')
    
    # subplot 做标记
    from datetime import datetime
    import pandas as pd

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    data = pd.read_csv('.\\input\\julyedu\\spx.csv', index_col=0, parse_dates=True)
    spx = data['SPX']
    
    spx.plot(ax=ax, style='k-')
    
    crisis_data = [
        (datetime(2007, 10, 11), 'Peak of bull market'),
        (datetime(2008, 3, 12), 'Bear Stearns Fails'),
        (datetime(2008, 9, 15), 'Lehman Bankruptcy')
    ]
    
    for date, label in crisis_data:
        ax.annotate(label, xy=(date, spx.asof(date) + 100),
                    xytext=(date, spx.asof(date) + 250),
                    arrowprops=dict(facecolor='black'),
                    horizontalalignment='left', verticalalignment='top')
    
    # Zoom in on 2007-2010
    ax.set_xlim(['1/1/2007', '1/1/2011'])
    ax.set_ylim([600, 1800])
    
    ax.set_title('Important dates in 2008-2009 financial crisis')
    
