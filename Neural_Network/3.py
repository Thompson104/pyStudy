#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pylab import *
import  os,sys

def f1(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

def f2(t):
    return np.sin(2 * np.pi * t) * np.cos(3 * np.pi * t)

mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

t = np.arange(0,5,0.02)
plt.figure(figsize=(8,7),dpi=98)
p1 = plt.subplot(211)
p2 = plt.subplot(212)

p1.plot(t,f1(t))
p1.axis([0.0,5.01,-1.0,1.5])
p1.grid(True)
p1.set_title(u'a这里写的是中文')
p2.plot(t,f1(t))
plt.title(u'b这里写的是中文')
plt.show()