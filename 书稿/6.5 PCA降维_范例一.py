# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 08:38:38 2018

@author: Tim
"""
#%% 导入模块
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from  sklearn.decomposition import PCA
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
def print_pca_info(model):    
    print('components_:',model.components_)
    print()
    myprint('explained_variance_',model.explained_variance_)
    myprint('explained_variance_ratio_',model.explained_variance_ratio_)
    myprint('mean_',model.mean_)
    myprint('singular_values_:',model.singular_values_)
    myprint('n_samples_:',model.n_samples_)
    myprint('noise_variance_:',model.noise_variance_)
    return
#%% 主程序
if __name__ == '__main__':  
    # 加载数据
    X,y,target_names,feature_names = data_helper()
    # 对比svd_solver参数对降维的影响
    svd_solvers = ['auto', 'full', 'arpack', 'randomized']
    for svd_solver in svd_solvers:    
        model = PCA(n_components=2,svd_solver=svd_solver)
        model.fit(X)
        print('='*70)
        myprint('svd_solver',svd_solver)
        print_pca_info(model)
        
    # 绘图对比降维对数据的影响
    model = PCA(n_components=2)
    X_pca = model.fit_transform(X)
    fig,axs = plt.subplots(1,2)
    axs[0].scatter(X[:,0] , X[:,2],c=y)
    axs[0].set_title('原始数据',fontproperties=myfont)
    axs[0].set_xlabel(feature_names[0])
    axs[0].set_ylabel(feature_names[2])
    
    axs[1].scatter(X_pca[:,0] , X_pca[:,1],c=y)
    axs[1].set_title('PCA降维后的数据',fontproperties=myfont)
    axs[1].set_xlabel('X_pca(0)')
    axs[1].set_ylabel('X_pca(1)')
    fig.set_tight_layout(True)
    plt.show()