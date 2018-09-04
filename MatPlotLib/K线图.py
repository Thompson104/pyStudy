# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:15:48 2017
@author: Tim
"""

import numpy as np
import pandas as pd
from pandas_datareader  import data,wb
from matplotlib.finance import candlestick_ochl
from matplotlib.dates import bytespdate2num,num2date,date2num
import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
# 灰色背景等的一个样式
print(plt.style.available) #可用的
plt.style.use('fivethirtyeight') #fivethirtyeight,bmh,ggplot

loadfromnet = False
if loadfromnet == True:    
    start = datetime.datetime(2017, 5, 1)
    end = datetime.datetime(2017,12,18)
    # IBM公司股票价格
    sh = data.DataReader("INTC", 'yahoo', start, end)
    data = sh.values
    date = sh.index.values
    sh.to_csv('intc.csv',index=True,sep=',')
else:
    data = np.loadtxt('intc.csv',skiprows=1,delimiter=',',usecols=(1,2,3,4,5,6))
    date = np.loadtxt('intc.csv',
                      skiprows=1,
                      delimiter=',',
                      usecols=(0),
                      converters={0:bytespdate2num('%Y-%m-%d')}
                      )
    alldata = np.loadtxt('intc.csv',
                      skiprows=1,
                      delimiter=',',
                      usecols=(0,1,2,3,4,5,6),
                      converters={0:bytespdate2num('%Y-%m-%d')}
                      )
    #plt.plot_date(date,data[:,2],'-')

# 图形布局
left,width = 0.1,0.8
rect_vol = [left,0.1,width,0.3]
rect_main = [left,0.41,width,0.5]

fig = plt.figure()
# 交易量图
ax_vol = fig.add_axes(rect_vol)
ax_vol.fill_between(date,data[:,-1],
                    color='y')
# 将x轴的数据以date类型对待
ax_vol.xaxis_date()
# 设置x轴标签属性
plt.setp(ax_vol.get_xticklabels(),rotation=30,horizontalalignment='right')

# K线图
# date是（162，）必须reshape为（162，1）
candlestickData = np.hstack((date.reshape(date.shape[0],-1), data[:,[0,3,1,2]]))
#candlestickData = alldata[:,(0,1,4,2,3)]
ax_main = fig.add_axes(rect_main)
# 蜡烛图
candlestick_ochl(ax_main,
                 candlestickData,
                 width=0.6,
                 colorup='r',#上涨用红色
                 colordown='g' #下跌用绿色
                 )
# 去掉k线图的x坐标的标记
# ax_main.set_xticks([])
ax_main.axes.get_xaxis().set_visible(False)
#
# fig.text(0.5,0.95,'IBM公司股票K线图',font_properties= myfont)
ax_main.set_title('IBM公司股票K线图与交易量图',font_properties= myfont)
plt.show()