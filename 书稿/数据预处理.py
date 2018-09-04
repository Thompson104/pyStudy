# -*- coding: utf-8 -*-
"""
数据预处理
"""
import numpy as np
#%% 二值化编码
from sklearn.preprocessing import Binarizer
X = [
     [1,2,3,4,5],
     [5,4,3,2,1],
     [3,3,3,3,3],
     [1,1,1,1,1]
     ]
binarizer = Binarizer(threshold=2.5)
print('='*35)
print('二值化转换之后的X\n',binarizer.transform(X))

#%% 独热码编码
from sklearn.preprocessing import OneHotEncoder
X = [
     [1,2,3,4,5],
     [5,4,3,2,1],
     [3,3,3,3,3],
     [1,1,1,1,1]
     ]
encoder = OneHotEncoder(sparse=False,n_values='auto')
encoder.fit(X)
print('='*35)
print('激活特征 active_features_:\t',encoder.active_features_)
print('feature_indices_:\t',encoder.feature_indices_)
print('n_values:\t',encoder.n_values)
print('编码之后',encoder.transform([[1,2,3,4,5]]))

#%% min-max标准化
from sklearn.preprocessing import MinMaxScaler
X = np.array([
     [1,6,1,2,10],
     [2,6,3,2,7],
     [-3,7,5,6,4],
     [4,8,7,8,1]
     ],dtype='float64')
scaler = MinMaxScaler(feature_range=(3,8))
scaler.fit(X)
print('='*35)
print('转换之前的X：\n',X)
print('min_ is ',scaler.min_)
print('data_min_ is ',scaler.data_min_)
print('data_max_ is',scaler.data_max_)
print('转换之后的X：\n',scaler.transform(X))

#%% MaxAbsScaler标准化
# 将每个属性值除以该属性的绝对值中最大的
from sklearn.preprocessing import MaxAbsScaler
X = np.array([
     [1,6,1,2,10],
     [2,6,3,2,7],
     [-3,7,5,6,4],
     [4,8,7,8,1]
     ],dtype='float64')
scaler = MaxAbsScaler()
scaler.fit(X)
print('='*35)
print('转换之前的X：\n',X)
print('max_abs_ is ',scaler.max_abs_)
print('n_samples_seen_ is ',scaler.n_samples_seen_)
print('scale_ is',scaler.scale_)
print('转换之后的X：\n',scaler.transform(X))

#%% z-score标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
print('='*35)
print('转换之前的X：\n',X)
print('scale_ is ',scaler.scale_)
print('mean_ is',scaler.mean_)
print('var_ is',scaler.var_)
print('转换之后的X：\n',scaler.transform(X))

#%% 正则化
from sklearn.preprocessing import Normalizer
scaler = Normalizer(norm='l2')
scaler.fit(X)
print('='*35)
print('转换之前的X：\n',X)

print('转换之后的X：\n',scaler.transform(X))


