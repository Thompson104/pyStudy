import numpy as np
from sklearn import decomposition as dp
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import  io as spi
# ===============================================================
'''
10种苹果香精的拉曼（XJlaman_4_1.mat）和离子迁移谱（XJlizi_4_1.mat）数据，
每种香精只购买一个批次，分别于不同时间段采集了30个数据，总共就是10*30=300个拉曼数据和300个离子迁移谱数据，
数据格式为matlab格式，需要用matlab软件打开。
拉曼数据的列数为300列，1-30列为A香精的数据，31-60为B香精数据，依次类推，
离子迁移谱数据分别与拉曼数据一一对应，也是300列。拉曼数据的行数为2090，代表一张拉曼谱图采集了2090个点，
同理离子迁移谱的6000行代表采集了6000个点。
'''
def load_Laman():
    raw_data = spi.loadmat('./input/XJlaman_4_1.mat')
    x = raw_data['lamanxiangjing']
    y = np.zeros((x.T.shape[0]))
    return x.T,y

[x1,y1] = load_Laman()
raw_data = spi.loadmat('./input/XJlaman_4_1.mat')
x = raw_data['lamanxiangjing']
# 取两个样品的数据
xx = x.T[0:60,:]
yy = np.zeros((60,1))
yy[30:,:] = 1
# ===============================================================
# 作图，观察数据
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in np.arange(0,10):
    xs = np.arange(2090)
    zs = xx[i]
    ax.plot(xs,np.linspace(i,i,2090), zs=zs,zdir='z',alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
# ===============================================================
#
pca = dp.PCA(n_components=3)
pca.fit(xx)
xx_pca = pca.transform(xx)
# ===============================================================
# 作图，观察数据
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xx_pca[0:60, 0], xx_pca[0:60, 1], xx_pca[0:60, 2],s=40,c= yy)
plt.show()
# ===============================================================
x_train,x_test,y_train,y_test = train_test_split(xx_pca,yy,random_state=2,test_size=0.1)
# ===============================================================
clf = svm.SVC(C=1.0,kernel='poly')
clf.fit(x_train,y_train)
results = clf.predict(x_test)
print(results)
print('# ===============================================================')
print(y_test.T)

