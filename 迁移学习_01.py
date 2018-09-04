import numpy as np
import scipy.io as spi
import pandas as pd
from sklearn import svm as svm
from sklearn.svm import SVC

raw_data = spi.loadmat('./input/XJlaman_4_1.mat')
x = raw_data['lamanxiangjing']
# 取两个样品的数据
xx = x.T[0:60,:]
yy = np.zeros((60,1))
yy[30:,:] = 1

model = svm.libsvm.fit(X=xx,Y=yy.reshape(60),svm_type=0,kernel='rbf')