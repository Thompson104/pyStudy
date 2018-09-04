import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from appleEssenceData import  *
import os
from sklearn import datasets

# os.chdir('D:\\20_同步文件\\pyStudy\\Apple_essence')
[raw_x,raw_y] = load_Laman()
# [raw_x,raw_y] = datasets.load_iris(return_X_y=True)

train_x, test_x, train_y, test_y = train_test_split(raw_x, raw_y, train_size=0.6, random_state=5)

model = SVC()
model.fit(train_x,train_y)
result1 = cross_val_score(model,test_x,test_y)
print(result1)

# model = SVC()
# 对负数进行处理，log计算不能x要大于0
minux = np.abs( np.min(raw_x) ) + 0.1
train_x1 = np.log2(train_x + minux)
test_x1 = np.log2(test_x + minux)
model.fit(train_x1,train_y)
result2 = cross_val_score(model,test_x1,test_y)
print(result2)

print('''
      进行简单的log数据处理，分类的准确性有所提高
      ''')