# -*- coding: utf-8 -*-
# 引入基础支持模块
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
myfont_title = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',
                                 size=16)
# 引入sklearn的相关模块与函数
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import datasets  as ds
from sklearn.svm import SVR,NuSVR
# 引入回归评价的指标
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
#%% 加载数据
def make_regression_data():
    X = np.linspace( -2 * np.pi, 2 * np.pi, 100)
    y = (3 * np.sin(X) + 2 * X + 2).ravel()
    # 加入噪音
    y[::7] += 7  * ( 0.5 - np.random.rand( 15 ) )
    # 切分数据
    X = X.reshape(-1,1)
    y = y.reshape(-1, 1)
    x_train,x_test,y_train,y_test = train_test_split(X,y,
                                                     random_state=0,
                                                     test_size=0.3,
                                                     shuffle=False
                                                     )
    
    return x_train,x_test,y_train,y_test,X,y  
#%% 输出回归器的各项回归指标
def print_regression_metrics(title,y_true,y_pred):
    print('='*40)
    print(title)
    
     # 输出分类器的在测试集上的性能报告  
    print('explained_variance_score:\t',explained_variance_score(y_true,y_pred))
    print('mean_absolute_error:\t\t',mean_absolute_error(y_true,y_pred))
    print('mean_squared_error:\t\t',mean_squared_error(y_true,y_pred))
    print('median_absolute_error:\t\t',median_absolute_error(y_true,y_pred))
    print('r2_score:\t\t\t',r2_score(y_true,y_pred))
    return    
#%% 测试SVR的不同核函数的回归效果
def compare_SVM_regression_kernel():    
    # 加载数据
    x_train,x_test,y_train,y_test,X,y = make_regression_data()
    # 构建回归模型
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=3)
    svr_sigmoid = SVR(kernel='sigmoid')
    # 计算预测值
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)
    y_sigmoid = svr_sigmoid.fit(X, y).predict(X)
    
    #输出在测试集上的性能
    print_regression_metrics('径向基核函数回归器的性能报告',
                             y.reshape(-1,1),
                             y_rbf.reshape(-1,1) )
    
    
    # 绘图
    fig = plt.figure(figsize=(9,9))
    lw = 2
    # 绘制rbf核函数的回归效果图
    ax = fig.add_subplot(221)    
    ax.scatter(X, y, color='darkorange', label='data')
    ax.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('径向基核函数',fontproperties=myfont)
    ax.legend(loc='best')
    
    # 绘制线性核函数的回归效果图
    ax = fig.add_subplot(222)
    ax.scatter(X, y, color='darkorange', label='data')
    ax.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('线性核函数',fontproperties=myfont)
    ax.legend(loc='best')
    
    # 绘制多项式核函数的回归效果图
    ax = fig.add_subplot(223)
    ax.scatter(X, y, color='darkorange', label='data')
    ax.plot(X, y_poly, color='y', lw=lw, label='Polynomial model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('多项式核函数',fontproperties=myfont)
    ax.legend(loc='best')
    
    # 绘制sigmoid核函数的回归效果图
    ax = fig.add_subplot(224)
    ax.scatter(X, y, color='darkorange', label='data')
    ax.plot(X, y_sigmoid, color='y', lw=lw, label='Sigmoid model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Sigmoid核函数',fontproperties=myfont)
    ax.legend(loc='best')
    
    plt.suptitle('不同核函数的支持向量回归效果对比',fontproperties=myfont_title)    
    plt.legend()
    plt.show()
    
    return 
#%% 测试SVR中参数C对回归效果的影响
def compare_SVR_C():
    x_train,x_test,y_train,y_test,X,y = make_regression_data()
    # 参数C的范围
    C_range = np.logspace(-2,3)
    train_scores = []
    test_scores = []
    for c in C_range:
        reg = SVR(C=c,kernel='rbf')
        reg.fit(x_train,y_train)

        train_scores.append( reg.score(x_train,y_train ) )
        test_scores.append( reg.score(x_test,y_test ) )        
        pass
    #
    index = C_range
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111)
    ax.plot(index,train_scores,label='training score')
    ax.plot(index,test_scores,label='testing score')
    ax.legend(loc='best')
    ax.set_xscale('log')
    #
    plt.show()
    return    


#%%
if __name__ == '__main__':
#    compare_SVR_C()
    compare_SVM_regression_kernel()
    