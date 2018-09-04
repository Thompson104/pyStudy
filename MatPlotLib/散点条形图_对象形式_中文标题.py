# -*- coding: utf-8 -*-
"""
将散点图与条形图结合
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
plt.style.use('ggplot')
np.random.seed(0)

x = np.random.randn(200)
y = x + np.random.randn(200) * 0.5

margin_border = 0.1
width  = 0.6
margin_between = 0.02 # 要小于0.1
height = 0.2

left_s = margin_border
bottom_s= margin_border
height_s = width
width_s = width

left_x = margin_border
bottom_x = margin_border + width + margin_between
height_x = height
width_x = width

left_y = margin_border + width + margin_between
bottom_y = margin_border
height_y = width
width_y = height

fig = plt.figure(1,figsize=(8,8))

rect_s = [left_s,bottom_s,width_s,height_s]
rect_x = [left_x,bottom_x,width_x,height_x]
rect_y = [left_y,bottom_y,width_y,height_y]

axScatter = fig.add_axes(rect_s)
axHisX = fig.add_axes(rect_x)
axHisX.set_xticks([])
axHisY = fig.add_axes(rect_y)
axHisY.set_yticks([])

axScatter.scatter(x,y)

bin_width = 0.15
xymax = np.max( [ np.max(np.fabs(x)),
                 np.max(np.fabs(y)) ] ) * bin_width
lim = int(xymax / bin_width +1)
axScatter.set_xlim(-lim,lim)
axScatter.set_ylim(-lim,lim)

axHisX.hist(x,bins=np.arange(-lim,lim+bin_width,bin_width))
axHisY.hist(y,bins=np.arange(-lim,lim+bin_width,bin_width) ,orientation='horizontal')

axHisX.set_xlim(axScatter.get_xlim())
axHisY.set_ylim(axScatter.get_ylim())

fig.text(0.5,0.95,'散点图与条形图结合',font_properties= myfont)
plt.show()

