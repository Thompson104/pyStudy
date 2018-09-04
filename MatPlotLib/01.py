# -*- coding: utf-8 -*-
"""
简单作图demo
"""

from matplotlib import pyplot as plt
import numpy as np
x = np.linspace(-1,1,50)
y1 = 2*x + 1
y2 = x**2
#创建第一幅图
plt.figure(num=2,figsize=(5,4))
fg = plt.plot(x,y1)
#创建第二幅图
plt.figure(num=5)
# 显示两个线条，在同一个figure上
plt.plot(x,y1,color='red',linewidth=15.0,linestyle='--')
plt.plot(x,y2)

#显示图片
plt.show()