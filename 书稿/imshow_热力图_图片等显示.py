# -*- coding: utf-8 -*-
"""
imshow()函数格式为：
matplotlib.pyplot.imshow(X, cmap=None)
X: 要绘制的图像或数组。
cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。

其它可选的颜色图谱如下列表：
颜色图谱	描述
autumn	红-橙-黄
bone	黑-白，x线
cool	青-洋红
copper	黑-铜
flag	红-白-蓝-黑
gray	黑-白
hot	黑-红-黄-白
hsv	hsv颜色空间， 红-黄-绿-青-蓝-洋红-红
inferno	黑-红-黄
jet	蓝-青-黄-红
magma	黑-红-白
pink	黑-粉-白
plasma	绿-红-黄
prism	 红-黄-绿-蓝-紫-...-绿模式
spring	洋红-黄
summer	绿-黄
viridis	蓝-绿-黄
winter	蓝-绿
用的比较多的有gray,jet等，如：
===============================================
热图（heatmap）是数据分析的常用方法，通过色差、亮度来展示数据的差异、易于理解。
Python在Matplotlib库中，调用imshow()函数实现热图绘制。

其中，X变量存储图像，可以是浮点型数组、unit8数组以及PIL图像，如果其为数组，则需满足一下形状：
    (1) M*N      此时数组必须为浮点型，其中值为该坐标的灰度；
    (2) M*N*3  RGB（浮点型或者unit8类型）
    (3) M*N*4  RGBA（浮点型或者unit8类型）

Colormap：参数cmap用于设置热图的Colormap。（参考百度百科）
Colormap是MATLAB里面用来设定和获取当前色图的函数，可以设置如下色图：
 hot 从黑平滑过度到红、橙色和黄色的背景色，然后到白色。
    cool 包含青绿色和品红色的阴影色。从青绿色平滑变化到品红色。
    gray 返回线性灰度色图。
    bone 具有较高的蓝色成分的灰度色图。该色图用于对灰度图添加电子的视图。
    white 全白的单色色图。 
    spring 包含品红和黄的阴影颜色。 
    summer 包含绿和黄的阴影颜色。
    autumn 从红色平滑变化到橙色，然后到黄色。 
    winter 包含蓝和绿的阴影色。
"""
from skimage import io,data
import matplotlib.pyplot as plt
x = [[1,2],[3,4],[5,6]]

fig = plt.figure(figsize=(10,8))
#%%
ax = fig.add_subplot(131)
plt.imshow(x)
# Colorbar：增加颜色类标
plt.colorbar(shrink=0.7)
#%%
ax = fig.add_subplot(132)
plt.imshow(x,cmap=plt.get_cmap('hot'),interpolation='nearest',vmin=1,vmax=6)
plt.colorbar(shrink=0.5)
#%%
ax = fig.add_subplot(133)
plt.imshow(x,cmap=plt.cm.get_cmap('gray'),interpolation='nearest',vmin=1,vmax=6)
plt.colorbar(shrink=0.5)

# 显示照片
plt.show()
plt.figure()
img =data.astronaut()
plt.imshow(img)
plt.show()