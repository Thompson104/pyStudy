# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 08:38:38 2018

@author: Tim
"""
#%% 导入模块
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from  sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
#%% 加载数据
def data_helper():
    raw_data = ds.load_iris()
    X = raw_data.data
    y = raw_data.target
    target_names = raw_data.target_names
    feature_names = raw_data.feature_names

    return X,y,target_names,feature_names

#%% 输出模型的参数
def myprint(name,value)  :
    print('%-25s : %s'%(name,value))
    return
def print_FactorAnalysis_info(model):    
    print('components_:',model.components_)
    print()
    myprint('loglike_',model.loglike_)
    myprint('noise_variance_',model.noise_variance_)
    myprint('n_iter_',model.n_iter_)
    return
#%% 主程序
if __name__ == '__main__':  
    # 加载数据
    X,y,target_names,feature_names = data_helper()
    # 对比svd_method参数对降维的影响
    svd_methods = ['lapack','randomized']
    for svd_method in svd_methods:    
        model = FactorAnalysis(n_components=2,svd_method=svd_method)
        model.fit(X)
        print('='*70)
        myprint('svd_method',svd_method)
        print_FactorAnalysis_info(model)
        
    # 绘图对比降维对数据的影响
    model = FactorAnalysis(n_components=2,copy=False)
    a = X.copy()
    X_fas = model.fit_transform(X)
    fig,axs = plt.subplots(1,2)
    axs[0].scatter(X[:,0] , X[:,2],c=y)
    axs[0].set_title('原始数据',fontproperties=myfont)
    axs[0].set_xlabel(feature_names[0])
    axs[0].set_ylabel(feature_names[2])
    
    axs[1].scatter(X_fas[:,0] , X_fas[:,1],c=y)
    axs[1].set_title('FactorAnalysis降维后的数据',fontproperties=myfont)
    axs[1].set_xlabel('X_pca(0)')
    axs[1].set_ylabel('X_pca(1)')
    fig.set_tight_layout(True)
    plt.show()