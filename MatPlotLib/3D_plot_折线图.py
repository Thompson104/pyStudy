from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spi

raw_data = spi.loadmat('../input/XJlaman_4_1.mat')
x = raw_data['lamanxiangjing']
# 取样品的数据
zz = x.T[:,:]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in np.arange(0,10):
    xs = np.arange(2090)
    zs = zz[i]
    # 特别要注意zdir的方向
    ax.plot(xs,np.linspace(i,i,2090), zs=zs,zdir='z',alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()