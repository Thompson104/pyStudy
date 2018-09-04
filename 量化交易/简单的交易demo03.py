# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:36:53 2018

@author: smart
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import talib as ta
import tushare as ts
import datetime
import mpl_finance as mpf

hs300 = ts.get_hist_data('hs300')
hist_data = hs300[['open','high','low','close','volume']]

data_list = []
data_volume = []

for dates,row in hist_data.iterrows():
    date_time = datetime.datetime.strptime(dates,'%Y-%m-%d')
    t = plt.date2num(date_time)
    open,high,low,close,volume= row
    datas = (t,open,high,low,close)
    data_v =(t,volume)
    data_list.append(datas)
    data_volume.append(data_v)

fig,(ax1,ax2) = plt.subplots(2,sharex=True,figsize=(12,6))
mpf.candlestick_ohlc(ax1,data_list,width=1.5,colorup='r',colordown='green')
ax1.set_title('hs300 index')
ax1.set_ylabel('Price')
ax1.grid(True)
ax1.xaxis_date()

#
volume = np.array(data_volume)
plt.bar(volume[:,0],volume[:,1]/10000)
ax2.grid(True)
ax2.autoscale_view() # 自动调整坐标轴范围
plt.setp(plt.gca().get_xticklabels(),rotation = 30)
