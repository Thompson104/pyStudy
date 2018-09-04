# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

#%% 生成矩阵
x = np.linspace(0,5,50)
y = np.linspace(0,6,60)
X,Y = np.meshgrid(x,y)
Z = np.sin(X) * np.cos(Y)

fig1 = plt.figure()
ax1 = plt.subplot(221, projection='3d')
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
ax1.set_title('原始图形',fontproperties=myfont)
plt.show()

X_t = X.reshape(1,-1).T
Y_t = Y.reshape(1,-1).T
Z_t = Z.reshape(1,-1).T

temp = np.concatenate((X_t,Y_t,Z_t),axis=1)
#%%
t = np.array([[1,2,3,4],
           [5,6,7,8],
           [9,10,11,12],
           [13,14,15,16]])

#%% 将图像向x-y投影
t = np.array([[1,0,0],
              [0,1,0],
              [0,0,0]])
temp_t = np.dot(temp,t)


ax2 = plt.subplot(222, projection='3d')
ax2.plot_surface(temp_t[:,0].reshape(X.shape), 
                temp_t[:,1].reshape(X.shape), 
                temp_t[:,2].reshape(X.shape), rstride=1, cstride=1)

ax2.set_zlim(ax1.get_zlim())
ax2.set_title('将图像向x-y投影',fontproperties=myfont)
plt.show()
#%% 以一个非零数k乘矩阵的某一行（列）：
'''
即对矩阵中某一向量进行伸缩变换，整个矩阵代表的图形对应发生变化，由于k不能为0，
所以矩阵张成空间的维数（秩）不变，方阵张成的平行几何体的空间积（行列式）变成原来的k倍
下面变换矩阵作用下，图形的Z坐标方向被缩小为原来的一半
'''
t = np.array([[1,0,0],
              [0,1,0],
              [0,0,0.5]])
temp_t = np.dot(temp,t)

#fig3 = plt.figure()
ax3 = plt.subplot(223, projection='3d')
ax3.plot_surface(temp_t[:,0].reshape(X.shape), 
                temp_t[:,1].reshape(X.shape), 
                temp_t[:,2].reshape(X.shape), 
                shade=True,rstride=1, cstride=1, cmap='rainbow')
ax3.set_zlim(ax1.get_zlim())
ax3.set_title('图形在Z轴方向进行压缩，压缩比例维0.5',fontproperties=myfont)
plt.show()

#%% 把矩阵的某一行（列）的k倍加于另一行（列）上
'''
对矩阵中某一向量做线性叠加，且新向量终点总是在另一向量的平行线上，所以对任意矩阵，图形产生了剪切变形，
由于剪切变形不会使向量重叠或缩为0，所以张成空间的维数也不变。
（注意观察图形变换前后的y向坐标值）
'''
t = np.array([[1,2,0],
              [0,1,0],
              [0,0,1]])
temp_t = np.dot(temp,t)

ax4 = plt.subplot(224, projection='3d')
ax4.plot_surface(temp_t[:,0].reshape(X.shape), 
                temp_t[:,1].reshape(X.shape), 
                temp_t[:,2].reshape(X.shape), 
                shade=True,rstride=1, cstride=1, cmap='rainbow')
ax4.set_zlim(ax1.get_zlim())
ax4.set_title('图形进行叠加，图形产生了剪切变形',fontproperties=myfont)
plt.show()

#%% 特征向量
x = np.array([[1,1,1,1],
              [1,1,-1,-1],
              [1,-1,1,-1],
              [1,-1,-1,1]])
a,b = np.linalg.eig(x)
print(x* b)
print(a*b)
x = np.diag((1, 2, 3))
w, v = np.linalg.eig(x)
print(x*v)
print(w*v)

#%% 矩阵的轶
# 生成5*6矩阵，轶维2
a = np.linspace(start=1,stop=30,num=30).reshape((3,-1))
print(np.linalg.matrix_rank(a))
b = np.array([[3,6,8,1,5,6,7,5,3,7],
              [1,2,3,4,5,6,7,8,9,10],
              [8,2,3,4,1,6,7,0,9,5]]).reshape(10,-1)
am= np.mat(a)
bm = np.mat(b)
cm = am*bm
print(np.linalg.matrix_rank(cm))
#%% 绘制范数图
#  L0与L1范数单位图
plt.figure()
plt.plot([-1.2,1.2],[0,0],'b-')
plt.plot([0,0],[-1.2,1.2],'b-')
plt.plot([1,0,-1,0,1],[0,1,0,-1,0],'g-',label='L1')
theta = np.arange(0, 2 * np.pi + 0.1,2 * np.pi / 1000)
x = np.cos(theta)
y = np.sin(theta)
plt.plot(x,y,'y-',label='L2')
plt.xlim([-2,2])
plt.ylim([-1.5,1.5])
plt.legend(loc='best')
plt.show()