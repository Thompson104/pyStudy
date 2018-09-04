# -*- coding: utf-8 -*-
"""
分类
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

def load_data():
    iris = ds.load_iris()
    x_train = iris.data
    y_train = iris.target
    # 分层采样
    return train_test_split(x_train,y_train,test_size=0.2,random_state=0,stratify=y_train)

def test_logisticRegression(*data):
    '''
    逻辑回归分类
    '''
    x_train,x_test,y_train,y_test = data
    regr = LinearRegression()
    regr.fit(x_train,y_train)
    print('系数 Coefficients:%s,\n 截距 Intercept %s'%(regr.coef_,regr.intercept_))
    print("score=",regr.score(x_test,y_test))
    return

def test_LinearDiscriminantAnalysis(*data):
    '''
    线性判别分析分类
    '''
    x_train,x_test,y_train,y_test = data
    regr = LinearDiscriminantAnalysis()
    regr.fit(x_train,y_train)
    print('系数 Coefficients:%s,\n 截距 Intercept %s'%(regr.coef_,regr.intercept_))
    print("score=",regr.score(x_test,y_test))
    return

def plot_LDA(convert_x,y):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors='rgb'
    markers='o*s'
    for target,color,marker in zip([0,1,2],colors,markers):
        pos = (y==target).ravel() # flattern 转为一维数组
        x = convert_x[pos,:]
        ax.scatter(x[:,0],x[:,1],x[:,2],color=color,marker=marker)
    ax.legend(loc='best')
    fig.suptitle('LDA 处理之后的 Iris 数据',font_properties= myfont)
    plt.show()
    return

if __name__ == '__main__':
    x_train,x_test,y_train,y_test = load_data()
    test_logisticRegression(x_train,x_test,y_train,y_test)
    test_LinearDiscriminantAnalysis(x_train,x_test,y_train,y_test)
    
    x = np.vstack((x_train,x_test))
    y = np.vstack((y_train.reshape(y_train.size,1),y_test.reshape(y_test.size,1)))
    lda = LinearDiscriminantAnalysis()
    lda.fit(x,y)
    convert_x = np.dot(x,lda.coef_.T) + lda.intercept_
    plot_LDA(convert_x,y)

