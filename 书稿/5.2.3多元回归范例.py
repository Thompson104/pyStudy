# -*- coding: utf-8 -*-
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%% 模拟数据生成
def data_helper(n_samples=100):
    '''
    生成用于多元线性回归分析的数据
    样本点的特征X维度为5维，其中有效信息为3维，即有2维数据是冗余的
    模拟数据的截距为9
    输出y的维度也为1维
    '''
    (X,y,coef)=ds.make_regression(n_samples=n_samples,n_features=5,n_informative=3,
                       n_targets=1,bias=9,coef=True,random_state=5)
    return (X,y,coef)

(X,y,coef) = data_helper()
print('模拟数据的系数',coef)
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=5)
#%%观察数据
# 定义绘图辅助函数
def plt_helper(label,title):
    fig =plt.figure()
    ax = fig.add_subplot(111,label=label)
    ax.set_title(title,fontproperties=myfont)
    ax.set_xlabel('x 轴',fontproperties=myfont)
    ax.set_ylabel('y 轴',fontproperties=myfont)
#    ax.axis([155,180,55,68])
    ax.grid(True)
    return ax
ax1 = plt_helper('ax1','观察模拟数据的分布')
ax1.plot(X[:,4],y,'r*')
#%%
linear_rg = LinearRegression(fit_intercept=True,#计算截距
                             normalize=False,   #回归之前不对数据集进行规范化处理
                             copy_X=True,       #复制X，不会对X的原始值产生影响
                             n_jobs=-1)         #使用所有的CPU

linear_rg.fit(train_x,train_y)
print("*"*15,'回归模型的参数',"*"*15)
print('回归模型的系数 = ',linear_rg.coef_)
print('回归模型的截距 = ',linear_rg.intercept_)

ax2 = plt_helper('ax2','观察测试数据的回归效果')
ax2.plot(test_x[:,4],linear_rg.predict(test_x),'r*',label='回归数据')
ax2.scatter(test_x[:,4],test_y,c='y',marker='o',s=100,label='原始数据')
ax2.legend(loc='best',prop=myfont)

#%%
print('回归模型的预测性能得分 =',linear_rg.score(test_x,test_y))