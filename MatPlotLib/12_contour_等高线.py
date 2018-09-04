import numpy as np
import matplotlib.pyplot as plt

# 计算高度的函数，不重要
def f(x,y):
    return (1-x/2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)
# cmap颜色映射表
plt.contourf(X,Y,f(X,Y),8,alpha=0.8,cmap=plt.cm.hot)
#18等高线的密度
C = plt.contour(X,Y,f(X,Y),8,colors='black',linewidth=0.5)
# 等高线的数据标签
plt.clabel(C,inline=True,fonesize=12)

plt.show()