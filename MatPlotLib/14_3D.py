import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d  import Axes3D

fig = plt.figure()
# 三维坐标
ax = Axes3D(fig)
x = np.arange(-4,4,.25)
y = np.arange(-4,4,0.25)
X,Y = np.meshgrid(x,y)
R = np.sqrt(X**2 + Y**2)
# 高度值
Z = np.sin(R)
# 在三维坐标值上画图
# rstride=1,cstride=1曲面上加上网格线,取值为跨度
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))

# zdir='z' 等高图的投影方向
ax.contourf(X,Y,Z,zdir='x',offset=-4,cmap='rainbow')
# z轴的取值范围
ax.set_zlim(-2,2)

plt.show()