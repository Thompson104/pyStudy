from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spi

raw_data = spi.loadmat('../input/XJlaman_4_1.mat')
x = raw_data['lamanxiangjing']
# 取两个样品的数据
yy = x.T[:,:]
xx = np.arange(1,2091)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in np.arange(0,4):
    xs = np.arange(2090)
    ys = yy[i]
    ax.bar(xs, ys, zs=i, zdir='y',alpha=0.8)
    # ax.plot(xs,ys,zs=z,zdir='y',alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()