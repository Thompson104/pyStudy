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
        y[i*30:i*30+30,]= i+1
    return x.T,y

os.chdir('D:\\20_同步文件\\pyStudy\\Apple_essence')
# [raw_x,raw_y] = load_Laman()
[raw_x,raw_y] = make_classification(n_samples=300,n_features=50,n_classes=2,
                                    n_clusters_per_class=2,n_informative=5,random_state=0)
n_components = 10
# pca = KernelPCA(n_components=n_components)
pca = PCA(n_components=n_components)
raw_x = pca.fit_transform(raw_x)

train_x,test_x,train_y,test_y = train_test_split(raw_x,raw_y,train_size=0.8,random_state=5)

model = svm.SVC(C=1,kernel='rbf')
model.fit(train_x,train_y)
predict_result = model.predict(test_x)
result = np.sum(predict_result == test_y) / test_y.shape[0]
print('准确率%0.2f %% '%(result*100)) #81.67%
