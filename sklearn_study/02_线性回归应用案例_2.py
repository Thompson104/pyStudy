import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import train_test_split #拆分数据集
from sklearn.cross_validation import cross_val_score #交叉验证
from sklearn.preprocessing import StandardScaler #归一化处理
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc',size=10)
# raw_data = spi.loadmat(r'../input/XJlaman_4_1.mat')

df = pd.read_csv(r'../input/winequality-red.csv',sep=';')

x = df[ list(df.columns)[:-1] ]# 不包括最后一列，即‘quality’
y = df['quality']

x_train,x_test,y_train,y_test = train_test_split(x,y)

# 归一化,x,y使用不同的归一化处理器
x_scaler = StandardScaler()
y_scaler = StandardScaler()
# 对训练集进行归一化训练和转换
x_train = x_scaler.fit_transform(x_train)
y_train = y_scaler.fit_transform(y_train)
# 对测试集进行归一化
x_test = x_scaler.transform(x_test)
y_test = y_scaler.transform(y_test)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor,x_train,y_train,cv=5)

print('交叉验证R方值:', scores)
print('交叉验证R方均值:', np.mean(scores))
regressor.fit_transform(x_train, y_train)
print('测试集R方值:', regressor.score(x_test, y_test))

'''
R方太小，说明该数据集是非线性的
'''