# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 21:10:50 2018

@author: TIM
"""

import matplotlib.pyplot as plt
import numpy as np
X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)
plt.figure(figsize=(8,4),dpi=100)
# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")
# 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")
# 设置横轴的上下限
plt.xlim(-4.0,4.0)
# 设置横轴记号
plt.xticks(np.linspace(-4,4,9,endpoint=True))
# 设置纵轴的上下限
plt.ylim(-1.0,1.0)
# 设置纵轴记号
plt.yticks(np.linspace(-1,1,5,endpoint=True))
# 以分辨率 100 来保存图片
plt.savefig("z:\\output.png",dpi=100)
# 在屏幕上显示
plt.show()
