import numpy as np
import pandas as pd
from sklearn import decomposition as dp
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  #归一化
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy import  io as spi
import time as tm
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
    '''加载拉曼谱图数据'''
    raw_data = spi.loadmat('./input/XJlaman_4_1.mat')
    x = raw_data['lamanxiangjing']
    y = np.zeros((x.T.shape[0]))
    for i in np.arange(0,30):
        y[i*30:i*30+30,]= i
    return x.T,y

def load_IMS():
    '''加载离子迁移谱数据'''
    raw_data = spi.loadmat('./input/XJlizi_4_1_1.mat')
    x = raw_data['XJlizi_4_1_1']
    y = np.zeros((x.T.shape[0]))
    for i in np.arange(0,30):
        y[i*30:i*30+30,]= i
    return x.T,y

[raw_x_ims,raw_y_ims] = load_IMS()
[raw_x_raman,raw_y_raman] = load_Laman()
# raw_x = np.concatenate((raw_x_raman,raw_x_ims),axis=1)
raw_x = raw_x_raman
raw_y = raw_y_ims

pca = PCA(n_components=10)
pca.fit(raw_x)
raw_x = pca.transform(raw_x)

# 数据预处理
kinds = 10  # 样本的种类
kindnum = 20    #每类训练样本的数量
nums = 30       #每类样本的数量
[n,m] = raw_x.shape
x_train = np.zeros((kindnum*kinds,m))
x_test  = np.zeros( ( (nums - kindnum) * kinds,m ) )
y_train = np.zeros((kindnum*kinds))
y_test = np.zeros( ( (nums-kindnum) * kinds ) )
# 选取训练数据
for i in np.arange(0,kinds):
    x_train[ i*kindnum : i*kindnum+kindnum,]= raw_x[ i * nums : i * nums + kindnum,]
    y_train[i * kindnum:i * kindnum + kindnum, ] = raw_y[i * nums:i * nums + kindnum, ]
    x_test[i* (nums - kindnum ):(i+1)*(nums - kindnum ),]= raw_x[i*nums+kindnum : (i+1)*nums,]
    y_test[i* (nums - kindnum ):(i+1)*(nums - kindnum ),] = raw_y[i*nums+kindnum : (i+1)*nums,]

# 归一化处理
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

#
from sklearn.model_selection import GridSearchCV
model = svm.SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1,2,3,4,5,6,7,8,9,10]}
t1 = tm.time()
clf = GridSearchCV(model,parameters)
clf.fit(raw_x,raw_y)
t2 = tm.time()
print('耗时：%f秒'%((t2-t1)))
sorted(clf.cv_results_.keys())
print(clf.best_params_)


model = svm.SVC(C=clf.best_params_['C'],kernel=clf.best_params_['kernel'])
model.fit(x_train,y_train)
y_test_1 = model.predict(x_test)
result = np.sum(y_test == y_test_1) / y_test_1.shape[0]
print('测试集准确率',result) # 79%

y_train_1 = model.predict(x_train)
result = np.sum(y_train == y_train_1) / y_train_1.shape[0]
print('训练集准确率',result) # 100%

from sklearn.metrics import roc_curve,roc_auc_score,auc
