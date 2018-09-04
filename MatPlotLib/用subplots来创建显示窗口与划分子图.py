# -*- coding: utf-8 -*-
'''
用subplots来创建显示窗口与划分子图
'''
import matplotlib.pyplot as plt
from skimage import data,color

img = data.immunohistochemistry()
hsv = color.rgb2hsv(img)
'''
直接用subplots()函数来创建并划分窗口。注意，比前面的subplot()函数多了一个s，该函数格式为：
matplotlib.pyplot.subplots(nrows=1, ncols=1)
nrows: 所有子图行数，默认为1。
ncols: 所有子图列数，默认为1。
返回一个窗口figure, 和一个tuple型的ax对象，该对象包含所有的子图,可结合ravel()函数列出所有子图，如：
'''
# 创建了2行2列4个子图，分别取名为ax0,ax1,ax2和ax3, 每个子图的标题用set_title()函数来设置
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
ax0, ax1, ax2, ax3 = axes.ravel()

ax0.imshow(img)
ax0.set_title("Original image")

ax1.imshow(hsv[:, :, 0], cmap=plt.cm.gray)
ax1.set_title("H")

ax2.imshow(hsv[:, :, 1], cmap=plt.cm.gray)
ax2.set_title("S")

ax3.imshow(hsv[:, :, 2], cmap=plt.cm.gray)
ax3.set_title("V")

for ax in axes.ravel():
    ax.axis('off')
'''
如果有多个子图，我们还可以使用tight_layout()函数来调整显示的布局，该函数格式为：
matplotlib.pyplot.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
所有的参数都是可选的，调用该函数时可省略所有的参数。
pad: 主窗口边缘和子图边缘间的间距，默认为1.08
h_pad, w_pad: 子图边缘之间的间距，默认为 pad_inches
rect: 一个矩形区域，如果设置这个值，则将所有的子图调整到这个矩形区域内。
'''
fig.tight_layout()  #自动调整subplot间的参数

#%% 
'''
除了使用matplotlib库来绘制图片，skimage还有另一个子模块viewer，
也提供一个函数来显示图片。不同的是，它利用Qt工具来创建一块画布，从而在画布上绘制图像。
'''
from skimage import data
from skimage.viewer import ImageViewer

img = data.coins()
viewer = ImageViewer(img)
viewer.show()


