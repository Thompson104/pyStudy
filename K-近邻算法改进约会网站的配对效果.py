# -*- coding: utf-8 -*-
"""
K-近邻算法改进约会网站的配对效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import MinMaxScaler

# 1、加载数据
all_df = pd.read_table('./input/datingTestSet.txt',header=None,names=['a','b','c','d'])

# 2、分析数据
all_df['d'] =  pd.Categorical(all_df['d']).codes  
plt.scatter(all_df['a'],all_df['b'],c=15 * all_df['d'])

# 3、准备数据：归一化数值
#scores = MinMaxScaler.fit(all_df.iloc[:,0:3].values.tolist())
all_df.iloc[:,0:3] = (all_df.iloc[:,0:3] - all_df.iloc[:,0:3].min())/ \
                (all_df.iloc[:,0:3].max() - all_df.iloc[:,0:3].min())

# 4、划分训练集与测试集
x_train,x_test,y_train,y_test = train_test_split(all_df[['a','b','c']],all_df['d'],random_state = 1,train_size=0.9)

# 4、建立模型
cls = KNeighborsClassifier(n_neighbors=5)
cls.fit( x_train , y_train)

# 5、验证模型正确率
predict = cls.predict(x_test.iloc[:,:])
predict1 = cls.predict_proba(x_test.iloc[:,:]) #返回属于类别的概率
#正确率
scores = cls.score(x_test.iloc[:,:],y_test.tolist())
a = predict - y_test.tolist()
print('正确率：%d %%'%(a.shape[0] - np.count_nonzero(a)) )


