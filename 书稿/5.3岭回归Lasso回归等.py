# -*- coding: utf-8 -*-
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
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
linear_rg = LinearRegression(fit_intercept=True,#计算截距
                             normalize=False,   #回归之前不对数据集进行规范化处理
                             copy_X=True,       #复制X，不会对X的原始值产生影响
                             n_jobs=-1)         #使用所有的CPU
linear_rg.fit(train_x,train_y)
y_rg = linear_rg.predict(test_x)

linear_Ridge = Ridge(alpha=0.0,fit_intercept=True,#计算截距
                             normalize=False,   #回归之前不对数据集进行规范化处理
                             copy_X=True       #复制X，不会对X的原始值产生影响
                             )  
linear_Ridge.fit(train_x,train_y)
y_Ridge = linear_Ridge.predict(test_x)

linear_Lasso = Lasso(alpha=1.0,fit_intercept=True,#计算截距
                             normalize=False,   #回归之前不对数据集进行规范化处理
                             copy_X=True       #复制X，不会对X的原始值产生影响
                             )  
linear_Lasso.fit(train_x,train_y)
y_Lasso = linear_Lasso.predict(test_x)

linear_ElasticNet = ElasticNet(alpha=1.0,fit_intercept=True,#计算截距
                             normalize=False,   #回归之前不对数据集进行规范化处理
                             copy_X=True       #复制X，不会对X的原始值产生影响
                             )
linear_ElasticNet.fit(train_x,train_y)
y_ElasticNet = linear_ElasticNet.predict(test_x)

print('模拟数据参数',coef)
print('线性回归模型参数',linear_rg.coef_)
print('岭回归模型参数',linear_Ridge.coef_)
print('Lasso回归模型参数',linear_Lasso.coef_)
print('ElasticNet回归模型参数',linear_ElasticNet.coef_)
#%%
ax2 = plt_helper('ax1','观察不同回归模型的效果')
ax2.plot(X[:,0],y,'r*',label="模拟数据")
ax2.plot(test_x[:,0], y_rg, '-k', label='线性回归模型')
ax2.plot(test_x[:,0], y_Ridge, '.b', label="岭回归模型")
ax2.plot(test_x[:,0], y_Lasso, '-g', label="Lasso归模型")
ax2.plot(test_x[:,0], y_ElasticNet, '-r', label="ElasticNet回归模型")

ax2.legend(loc='best',prop=myfont)
#%% 对不同的alpha值对预测性能进行检验
def test_Ridge_lasso_alpha(*data):
    train_x,test_x,train_y,test_y = data
    alphas = np.logspace(0,3,num=10)
    scores_Ridge = []
    scores_lasso = []
    scores_ElasticNet = []
    for i,alpha in enumerate(alphas):
        regr_ridge = Ridge(alpha=alpha)
        regr_ridge.fit(train_x,train_y)
        scores_Ridge.append(regr_ridge.score(test_x,test_y))
        
        regr_lasso = Lasso(alpha=alpha)
        regr_lasso.fit(train_x,train_y)
        scores_lasso.append(regr_lasso.score(test_x,test_y))
        
        regr_ElasticNet = ElasticNet(alpha=alpha)
        regr_ElasticNet.fit(train_x,train_y)
        scores_ElasticNet.append(regr_ElasticNet.score(test_x,test_y))
        
    ax3 = plt_helper('ax3','alph参数与回归性能',xlabel=r'$\alpha$取值',ylabel='归模型的预测性能')
    ax3.plot(alphas,scores_Ridge,label='岭回归')
    ax3.plot(alphas,scores_lasso,label='Lasso回归')
    ax3.plot(alphas,scores_ElasticNet,label='ElasticNet回归')
    ax3.legend(loc='best',prop=myfont)
    ax3.set_xscale('log')
    
    return

test_Ridge_lasso_alpha(train_x,test_x,train_y,test_y)

#%%
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def test_ElasticNet_alpha_beta(*data):
    train_x,test_x,train_y,test_y = data
    alphas = np.logspace(0,2)
    betas = np.linspace(0.01,1)
    scores_ElasticNet = []  
    for alpha in alphas:
        for beta in betas:
            regr_ElasticNet = ElasticNet(alpha=alpha,l1_ratio=beta)
            regr_ElasticNet.fit(train_x,train_y)
            scores_ElasticNet.append(regr_ElasticNet.score(test_x,test_y))   
    
    #绘图
    alphas1,betas1 = np.meshgrid(alphas,betas)
    scores = np.array( scores_ElasticNet ).reshape(alphas1.shape)

    fig= plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas1,betas1,scores,
                           rstride=1,cstride=1,cmap=cm.jet,antialiased=False)
    fig.colorbar(surf)
    ax.set_xlabel(r'$\alpha$',fontproperties=myfont)
    ax.set_ylabel(r'$\beta$',fontproperties=myfont)
    ax.set_zlabel(r'score',fontproperties=myfont)
    ax.set_title('ElasticNet回归',fontproperties=myfont)
    plt.show()
    return
test_ElasticNet_alpha_beta(train_x,test_x,train_y,test_y)
