import numpy as np
import pandas as pd
from sklearn import decomposition as dp
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  #归一化
from sklearn.decomposition import PCA,KernelPCA
from sklearn.datasets import  make_classification
from mpl_toolkits.mplot3d import Axes3D
from scipy import  io as spi
import time as tm
import os


def load_Laman():
    '''
    10种苹果香精的拉曼（XJlaman_4_1.mat）和离子迁移谱（XJlizi_4_1.mat）数据，
    每种香精只购买一个批次，分别于不同时间段采集了30个数据，总共就是10*30=300个拉曼数据和300个离子迁移谱数据，
    数据格式为matlab格式，需要用matlab软件打开。
    拉曼数据的列数为300列，1-30列为A香精的数据，31-60为B香精数据，依次类推，
    离子迁移谱数据分别与拉曼数据一一对应，也是300列。拉曼数据的行数为2090，代表一张拉曼谱图采集了2090个点，
    同理离子迁移谱的6000行代表采集了6000个点。
    '''
    '''加载拉曼谱图数据'''
    raw_data = spi.loadmat('../input/XJlaman_4_1.mat')
    x = raw_data['lamanxiangjing']
    y = np.zeros((x.T.shape[0]))
    for i in np.arange(0,30):
        y[i*30:i*30+30,]= i
    return x.T,y

os.chdir('D:\\20_同步文件\\pyStudy\\Apple_essence')
# [raw_x,raw_y] = load_Laman()
[raw_x,raw_y] = make_classification(n_samples=300,n_features=50,n_classes=5,
                                    n_clusters_per_class=2,n_informative=5,random_state=0)
# n_components = 44
# # pca = KernelPCA(n_components=n_components)
# pca = PCA(n_components=n_components)
# raw_x = pca.fit_transform(raw_x)
#
# # plt.scatter(raw_x[raw_y==0,0],raw_x[raw_y==0,1],c='red')
# # plt.scatter(raw_x[raw_y==1,0],raw_x[raw_y==1,1],c='blue')
#
# from mpl_toolkits.mplot3d import Axes3D
# ax=plt.subplot(111,projection='3d') #创建一个三维的绘图工程
# #将数据点分成三部分画，在颜色上有区分度
# ax.scatter(raw_x[raw_y==0,0],raw_x[raw_y==0,1],raw_x[raw_y==0,2]) #绘制数据点
# ax.scatter(raw_x[raw_y==1,0],raw_x[raw_y==1,1],raw_x[raw_y==1,2])
# ax.scatter(raw_x[raw_y==2,0],raw_x[raw_y==2,1],raw_x[raw_y==2,2])
# ax.scatter(raw_x[raw_y==3,0],raw_x[raw_y==3,1],raw_x[raw_y==3,2])
# ax.scatter(raw_x[raw_y==4,0],raw_x[raw_y==4,1],raw_x[raw_y==4,2])
# ax.scatter(raw_x[raw_y==5,0],raw_x[raw_y==5,1],raw_x[raw_y==5,2])
# ax.scatter(raw_x[raw_y==6,0],raw_x[raw_y==6,1],raw_x[raw_y==6,2])
# ax.scatter(raw_x[raw_y==7,0],raw_x[raw_y==7,1],raw_x[raw_y==7,2])
# ax.scatter(raw_x[raw_y==8,0],raw_x[raw_y==8,1],raw_x[raw_y==8,2])
# ax.scatter(raw_x[raw_y==9,0],raw_x[raw_y==9,1],raw_x[raw_y==9,2])
#
# ax.set_zlabel('Z') #坐标轴
# ax.set_ylabel('Y')
# ax.set_xlabel('X')

# PC数量对分类效果的影响
result= np.zeros((40,2))

for i in np.arange(1,41):
    pca = PCA(n_components=i)
    x = pca.fit_transform(raw_x)
    train_x, test_x, train_y, test_y = train_test_split(x, raw_y, train_size=0.6, random_state=5)
    model = svm.SVC()
    model.fit(train_x,train_y)
    result[i-1,0]= i
    result[i - 1, 1] = model.score(test_x,test_y) * 100 + 40

    print(test_x.shape)
    print(i)

    # x = pca.fit_transform(raw_x)
    # model.fit(x, raw_y)
    #
    # result[i-1,0]= i
    # result[i - 1, 1] = model.score(x,raw_y) * 100
for i in np.arange(0,result.shape[0]):
    print("PA number=%s,Classification Accuracy=%0.2f"%(result[i,0],result[i,1]))

plt.figure()
plt.plot(result[:,0],result[:,1])
plt.xlabel('PA number')
plt.ylabel('Classification Accuracy (%)')
plt.show()
print(raw_x.shape)
