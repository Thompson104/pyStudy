from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spi
'''
10种苹果香精的拉曼（XJlaman_4_1.mat）和离子迁移谱（XJlizi_4_1.mat）数据，
每种香精只购买一个批次，分别于不同时间段采集了30个数据，总共就是10*30=300个拉曼数据和300个离子迁移谱数据，
数据格式为matlab格式，需要用matlab软件打开。
拉曼数据的列数为300列，1-30列为A香精的数据，31-60为B香精数据，依次类推，
离子迁移谱数据分别与拉曼数据一一对应，也是300列。拉曼数据的行数为2090，代表一张拉曼谱图采集了2090个点，
同理离子迁移谱的6000行代表采集了6000个点。
'''
raw_data = spi.loadmat('../input/XJlaman_4_1.mat')
x = raw_data['lamanxiangjing']
# 取10个样品的数据
yy = x.T[:,:]
xx = np.arange(1,2091)

fig = plt.figure()
# ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
# yy = yy[1,61,91,121,151,181,211,241,271]
yy= yy[[1,61,91,121,151,181,211,241,271]]
for i in np.arange(0,9):
    xs = np.arange(2090)
    ys = yy[i]
    ax.bar(xs, ys, zs=i, zdir='y',alpha=0.8)
    # ax.plot(xs,ys,zs=z,zdir='y',alpha=0.8)



ax.set_xlabel('Ranman Shift(cm-1)')
ax.set_ylabel('Apple essences samples')
ax.set_zlabel('Raman Signal Intensity(a.u)')
ax.set_yticks([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1  ,9.1])
ax.set_yticklabels(('S','Q','f','e','d','c','b','a','l'))
# ax.set_yticklabels(([0,1,2,3,],['a','b','c','d']),minor=False)
plt.show()