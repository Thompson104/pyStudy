# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
import datetime
import matplotlib as mpl
from matplotlib import pyplot as plt

# x = np.arange(1,11,1)
# plt.plot(x,x)
# ax = plt.gca() # 获取当前坐标轴
# ax.locator_params('x',nbins=20) # 设置x轴坐标刻度的密度

# 面向对象的形式
fig = plt.figure()
# 生成日期序列
start = datetime.datetime(2017,1,1)
end = datetime.datetime(2017,12,31)
delta = datetime.timedelta(days=1)
# 准备数据
dates = mpl.dates.drange(start,end,delta)
y = np.random.randn(len(dates))
y1 = y + 100

plt.plot_date(dates,y,linestyle='-',marker='')
date_format = mpl.dates.DateFormatter('%Y-%m')
ax = plt.gca()
ax.xaxis.set_major_formatter(date_format)

fig.autofmt_xdate() # 自适应调整日期数据类型的标签

# =====================================
# 双坐标的关键,即公用x
plt.twinx()
# =====================================
plt.plot_date(dates,y1,color='r')

plt.show()

