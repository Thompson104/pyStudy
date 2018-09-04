# -*- coding: utf-8 -*-
import numpy as np #科学计算
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

from sklearn.tree import DecisionTreeRegressor,export_graphviz,export
from sklearn import datasets as ds 
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

import datetime
#%% 
def load_data():
    raw_data = ds.load_boston()
    X = raw_data.data
    y = raw_data.target
    feature_names = raw_data.feature_names
    # 设定根据y的值进行分层无偏采样
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                     shuffle=True,
                                                     random_state=0)
    return x_train,x_test,y_train,y_test,feature_names

#%% 使用DecisionTreeClassifier进行回归
def do_DecisionTreeRegressor(*data):
    x_train,x_test,y_train,y_test,feature_names = data
    reg = DecisionTreeRegressor()
    reg.fit(x_train,y_train)
   
    print_DecisionTreeRegressor_info(reg,
                                          '使用默认参数构建的决策树分类器',
                                          y_true=y_test,
                                          y_pred=reg.predict(x_test),
                                          feature_names=feature_names)
    return reg

#%% 考察不同切分准则对回归性能的影响
def compare_DecisionTreeRegressor_criterion(*data):
    begin_t = datetime.datetime.now()
    x_train,x_test,y_train,y_test,feature_names = data
    criterions = ['mse','mae','friedman_mse']
    regs = []
    for criterion in criterions:
        reg = DecisionTreeRegressor(criterion=criterion)
        reg.fit(x_train,y_train)
        print_DecisionTreeRegressor_info(reg,
                                          'criterion = %s'%(criterion),
                                          y_true=y_test,
                                          y_pred=reg.predict(x_test),
                                          feature_names=feature_names)
        regs.append(reg)
    end_t = datetime.datetime.now()
    print('耗时：',end_t - begin_t)
    return  regs     
#%% 考察随机划分与最优划分对回归效果的影响
def compare_DecisionTreeRegressor_splitter(*data):
    begin_t = datetime.datetime.now()
    x_train,x_test,y_train,y_test,feature_names = data
    splitters = ['best','random']
    regs = []
    for splitter in splitters:
        reg = DecisionTreeRegressor(splitter=splitter)
        reg.fit(x_train,y_train)
        print_DecisionTreeRegressor_info(reg,
                                          'splitter = %s'%(splitter),
                                          y_true=y_test,
                                          y_pred=reg.predict(x_test),
                                          feature_names=feature_names)
        regs.append(reg)
    end_t = datetime.datetime.now()
    print('耗时：',end_t - begin_t)
    return  regs
#%% 考察tree_depth对回归效果的影响
def compare_DecisionTreeRegressor_treeDepth(*data):
    begin_t = datetime.datetime.now()
    x_train,x_test,y_train,y_test,feature_names,max_depths = data
    max_depths = np.arange(3,max_depths)
    regs = []
    train_scores = []
    test_scores =[]
    for maxdepth in max_depths:
        reg = DecisionTreeRegressor(max_depth=maxdepth)
        reg.fit(x_train,y_train)
        # 保持训练集与测试集的预测准确率
        train_scores.append(reg.score(x_train,y_train))
        test_scores.append(reg.score(x_test,y_test))
        regs.append(reg)
    end_t = datetime.datetime.now()
    print('耗时：',end_t - begin_t)
    draw_compare_DecisionTreeRegressor_treeDepth(max_depths,train_scores,test_scores)
    return  regs
#%%
# 绘制不同深度的决策树的分类效果图   
def draw_compare_DecisionTreeRegressor_treeDepth(depths,train_scores,test_scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(depths,train_scores,'r-',label='训练集准确率')
    ax.plot(depths,test_scores,'b-',label='测试集准确率')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('准确率',fontproperties=myfont)
    ax.legend(loc='best',prop=myfont)
    plt.show()
    return

# 输出回归器的信息
def print_DecisionTreeRegressor_info(reg,title,y_true,y_pred,feature_names):
    # 
    print('='*35)
    print(title)
    print('各特征的重要性：')
    for feature_name,importance in zip(feature_names,reg.feature_importances_):
        print(feature_name,'\t=',importance)
    print('')
    print('max_features_ =\t',reg.max_features_)
    print('n_classes_ =\t',reg.n_classes_)
    print('n_features_ =\t',reg.n_features_)
    print('n_outputs_ =\t',reg.n_outputs_)
    
    # 输出分类器的在测试集上的性能报告    
    print('回归器的在测试集上的性能报告')
    print('explained_variance_score:\t',explained_variance_score(y_true,y_pred))
    print('mean_absolute_error:\t\t',mean_absolute_error(y_true,y_pred))
    print('mean_squared_error:\t\t',mean_squared_error(y_true,y_pred))
    print('mean_squared_log_error:\t\t',mean_squared_log_error(y_true,y_pred))
    print('median_absolute_error:\t\t',median_absolute_error(y_true,y_pred))
    print('r2_score:\t\t\t',r2_score(y_true,y_pred))
    
    return    
#%%
x_train,x_test,y_train,y_test,feature_names = load_data()
maxdepth=20
reg = do_DecisionTreeRegressor(x_train,x_test,y_train,y_test,feature_names)
regs_criterion = compare_DecisionTreeRegressor_criterion(x_train,
                                                         x_test,
                                                         y_train,
                                                         y_test,
                                                         feature_names) 

regs_splitter = compare_DecisionTreeRegressor_splitter(x_train,
                                                         x_test,
                                                         y_train,
                                                         y_test,
                                                         feature_names) 

regs_treeDepth = compare_DecisionTreeRegressor_treeDepth(x_train,
                                                         x_test,
                                                         y_train,
                                                         y_test,
                                                         feature_names,
                                                         maxdepth) 