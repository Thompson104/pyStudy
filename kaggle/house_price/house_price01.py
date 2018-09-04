# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 20:54:50 2017

@author: TIM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('./input/train.csv',index_col=0)
test_df = pd.read_csv('./input/test.csv',index_col=0)

# 浏览数据
train_df.head()

# ！观察SalePrice数据
train_df['SalePrice'].hist()
train_df['SalePrice'].describe()

# log处理让数据平滑化一些，加1是为了防止有0的情况，log1p即log(x+1)
# 反平滑化 
prices = pd.DataFrame({'price':train_df['SalePrice'],
                       'log(price + 1)':np.log1p(train_df['SalePrice'])})
prices.hist()

# 将SalePrice截出，赋值给y_train,这样train_df中仅仅保留特征值
y_train = np.log1p(train_df.pop('SalePrice'))

# ！将数据进行合并
all_df = pd.concat((train_df,test_df),axis=0)

# ！正则化变量属性:
#MSSubClass 的值其实应该是一个category，这种东西就很有误导性，我们需要把它变回成string
all_df['MSSubClass'].dtypes
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)

all_df['MSSubClass'].value_counts()

# 把category的变量转变成numerical表达形式的one-hot编码，独热码
pd.get_dummies(all_df['MSSubClass'],prefix='MSSubClass').head()

all_dummy_df = pd.get_dummies(all_df)


# ！处理缺失数据
# 观察数据缺失情况，缺失最多的column是LotFrontage
all_dummy_df.isnull().sum().sort_values(ascending=False).head(10)

# 用平均值来填满这些空缺
mean_cols = all_dummy_df.mean()
mean_cols.head(10)


all_dummy_df = all_dummy_df.fillna(mean_cols)

all_dummy_df.isnull().sum().sum() # 没有空缺了

# ！标准化numerical数据
'''
这一步并不是必要，但是得看你想要用的分类器是什么。
一般来说，regression的分类器都比较傲娇，最好是把源数据给放在一个标准分布内。
不要让数据间的差距太大。这里，我们当然不需要把One-Hot的那些0/1数据给标准化。
我们的目标应该是那些本来就是numerical的数据：
'''
# 先来看看 哪些是numerical的
numeric_cols = all_df.columns[all_df.dtypes != 'object']
'''
计算标准分布：(X-X')/s
让我们的数据点更平滑，更便于计算。
注意：我们这里也是可以继续使用Log的，我只是给大家展示一下多种“使数据平滑”的办法
'''
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

# !建立模型
# 把数据集分回 训练/测试集
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

# 用Ridge Regression模型来跑一遍看看。
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
# 这一步不是很必要，只是把DF转化成Numpy Array，这跟Sklearn更加配
X_train = dummy_train_df.values
X_test = dummy_test_df.values

# 用Sklearn自带的cross validation方法来测试模型
alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

# 大概alpha=10~20的时候，可以把score达到0.135左右
plt.plot(alphas, test_scores)
plt.title("Alpha vs CV Error")

# Random Forest方法
from sklearn.ensemble import RandomForestRegressor

max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    print('test_score=',test_score)
    test_scores.append(np.mean(test_score))

plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error")

# Step 5: Ensemble
'''
这里我们用一个Stacking的思维来汲取两种或者多种模型的优点
首先，我们把最好的parameter拿出来，做成我们最终的model
'''
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)

ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
'''
上面提到了，因为最前面我们给label做了个log(1+x), 于是这里我们需要把predit的值给exp回去，并且减掉那个"1"
所以就是我们的expm1()函数
'''
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))

# 一个正经的Ensemble是把这群model的预测结果作为新的input，
# 再做一次预测。这里我们简单的方法，就是直接『平均化』。
y_final = (y_ridge + y_rf) / 2

# Step 6: 提交结果
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})









