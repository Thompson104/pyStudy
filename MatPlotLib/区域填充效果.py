# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

x = np.linspace(0,5 * np.pi,1000)
y1 = np.sin(x)
y2 = np.sin(2*x)

plt.plot(x,y1,x,y2)
# # 填充图形与x轴之间的图形
# plt.fill(x,y1,'b')
# plt.fill(x,y2,'r',alpha=0.3)
# 填充图形之间的区域
plt.fill_between(x,y1,y2,where=y1<y2,facecolor = "r", interpolate= True)
plt.fill_between(x,y1,y2,where=y1>y2,facecolor = "b", interpolate= True)
plt.show()
