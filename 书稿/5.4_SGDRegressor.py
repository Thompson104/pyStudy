# -*- coding: utf-8 -*-
#%% 导入模块
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%% 模拟数据生成
def data_helper(n_samples=1000):
    '''
     产生数据样本点集合
     样本点的特征X维度为1维，输出y的维度也为1维
     输出是在输入的基础上加入了高斯噪声N（0,10）
     产生的样本点数目为1000个
    '''
    X, y, coef = ds.make_regression(n_samples=n_samples,
                                          n_features=1,
                                          n_informative=3,
                                          noise=10,
                                          coef=True,
                                          random_state=0)
    # 将上面产生的样本点中的前50个设为异常点（外点）
    # 即：让前50个点偏离原来的位置，模拟错误的测量带来的误差
    n_outliers = 100
    np.random.seed(0)
    X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
    y[:n_outliers] = -3 + 0.5 * np.random.normal(size=n_outliers)

    return (X,y,coef)

(X,y,coef) = data_helper()
print('模拟数据的系数',coef)
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=5)
#%%
#X_scaler = StandardScaler()
#y_scaler = StandardScaler()
#train_x = X_scaler.fit_transform(train_x.reshape(-1,1))
#train_y = y_scaler.fit_transform(train_y.reshape(-1,1))
#test_x = X_scaler.transform(test_x.reshape(-1,1))
#test_y = y_scaler.transform(test_y.reshape(-1,1))
#%%观察数据
# 定义绘图辅助函数
def plt_helper(label,title,xlabel='x 轴',ylabel='y 轴'):
    fig =plt.figure()
    ax = fig.add_subplot(111,label=label)
    ax.set_title(title,fontproperties=myfont)
    ax.set_xlabel(xlabel,fontproperties=myfont)
    ax.set_ylabel(ylabel,fontproperties=myfont)
    ax.grid(True)
    return ax
ax1 = plt_helper('ax1','观察模拟数据的分布')
ax1.plot(X[:,0],y,'r*')
#%%
linear_SGD = SGDRegressor(loss='squared_loss',max_iter=100)
linear_SGD.fit(train_x,train_y)
y_SGD = linear_SGD.predict(test_x)

linear_rg = LinearRegression(fit_intercept=True,#计算截距
                             normalize=False,   #回归之前不对数据集进行规范化处理
                             copy_X=True,       #复制X，不会对X的原始值产生影响
                             n_jobs=-1)         #使用所有的CPU
linear_rg.fit(train_x,train_y)
y_rg = linear_rg.predict(test_x)

print('模拟数据参数',coef)
print('SGDRegressor模型参数',linear_SGD.coef_)
print('LinearRegression模型参数',linear_rg.coef_)

scores = cross_val_score(linear_SGD, train_x, train_y, cv=5)
print('SGDRegressor交叉验证R方值:', scores)
print('SGDRegressor交叉验证R方均值:', np.mean(scores))
print('SGDRegressor测试集R方值:', linear_SGD.score(test_x, test_y))

scores = cross_val_score(linear_rg, train_x, train_y, cv=5)
print('LinearRegression交叉验证R方值:', scores)
print('LinearRegression交叉验证R方均值:', np.mean(scores))
print('LinearRegression测试集R方值:', linear_rg.score(test_x, test_y))

#%%
ax2 = plt_helper('ax1','观察不同回归模型的效果')
ax2.plot(X[:,0],y,'r*',label="模拟数据")
ax2.plot(test_x[:,0], y_SGD, '-k', label='SGDRegressor模型')
ax2.plot(test_x[:,0], y_rg, '-k', label='线性回归模型')
ax2.legend(loc='best',prop=myfont)
