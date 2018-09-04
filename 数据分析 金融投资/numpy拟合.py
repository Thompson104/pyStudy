import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return np.sin(x) + 0.5 * x

# 1、有噪音的数据
xn = np.linspace(-2 * np.pi, 2 * np.pi,50)
xn = xn + 0.15 * np.random.standard_normal(len(xn)) # 注意第二项的维度

yn = f(xn) + 0.25 * np.random.standard_normal(len(xn))

reg = np.polyfit(xn,yn,7)
ry = np.polyval(reg,xn)

plt.plot(xn,yn,'b-',label='f(x)')
plt.plot(xn,ry,'ro',label='regression')
plt.grid(True)
plt.legend(loc='best')

# 2、未排序数据
xn = np.random.rand(150) * 4 * np.pi -2 * np.pi
yn = f(xn)
print(xn[:10],'\n',yn[0:10])

reg = np.polyfit(xn,yn,7)
ry = np.polyval(reg,xn)

plt.plot(xn,yn,'b^',label='f(x)')
plt.plot(xn,ry,'ro',label='regression')
plt.grid(True)
plt.legend(loc='best')

# 3、多维数据
def fn((x,y)):
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2
x = np.linspace(0,10,20)
y = np.linspace(0,10,20)
X,Y = np.meshgrid(x,y)
Z = fn((X,Y))
x = X.flatten()
y = Y.flatten()
# 三维图,观察数据
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
fig = plt.figure(figsize=(9,6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,Z,rstride=2,cstride=2,
                      cmap=mpl.cm.coolwarm,linewidth=0.5,antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('fn(x,y)')
fig.colorbar(surf)
# 拟合,首先构造各维度的数据，依据np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2
matrix = np.zeros((len(x),6+1))
matrix[:,6] = np.sqrt(x)
matrix[:,5] = np.sin(x)
matrix[:,4] = y ** 2
matrix[:,3] = x ** 2
matrix[:,2] = y
matrix[:,1] = x
matrix[:,0] = 1

import statsmodels.api as sm
# 最小二乘方法来预测
model = sm.OLS(fn(x,y),matrix).fit()
model.rsquared
