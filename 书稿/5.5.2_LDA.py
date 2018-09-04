# -*- coding: utf-8 -*-
#%% 导入模块
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%% 使用iris数据集
def data_helper():
    '''
    shiyon
    '''
    data = ds.load_iris()
    
    return data

iris = data_helper()

train_x,test_x,train_y,test_y = train_test_split(iris.data,iris.target,
                                                 test_size=0.3,
                                                 stratify = iris.target,
                                                 random_state=5)

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
#以第3个索引为划分依据，x_index的值可以为0，1，2，3
x_index=3
ax1 = plt_helper('ax1','观察iris数据集的直方图',ylabel='',xlabel=iris.feature_names[x_index])
color=['blue','red','green']
for label,color in zip(range(len(iris.target_names)),color):
    ax1.hist(iris.data[iris.target==label,x_index],label=iris.target_names[label],color=color)

#ax1.set_xlabel(iris.feature_names[x_index])
ax1.legend(loc="best")

#画散点图，第一维的数据作为x轴和第二维的数据作为y轴
ax2 = plt_helper('ax1','观察iris数据集的散点图')
x_index=0
y_index=2
colors=['blue','red','green']
for label,color in zip(range(len(iris.target_names)),colors):
    ax2.scatter(iris.data[iris.target==label,x_index],
                iris.data[iris.target==label,y_index],
                label=iris.target_names[label],
                c=color)
ax2.set_xlabel(iris.feature_names[x_index])
ax2.set_ylabel(iris.feature_names[y_index])
ax2.legend(loc='best')

#%% 默认参数的分类效果
model = LinearDiscriminantAnalysis(n_components=2)
model.fit(train_x,train_y)
print('LDA模型的权重向量：%s,\n模型的截距：%s'%(model.coef_,model.intercept_))

score = model.score(test_x,test_y)
print('LDA模型的预测准确率：',score)










