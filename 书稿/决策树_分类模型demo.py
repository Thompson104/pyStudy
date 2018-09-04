# -*- coding: utf-8 -*-
import numpy as np #科学计算
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

from sklearn.tree import DecisionTreeClassifier,export_graphviz,export
from sklearn import datasets as ds 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import graphviz
import datetime
#%% 
def load_data():
    raw_data = ds.load_iris()
    X = raw_data.data
    y = raw_data.target
    target_names = raw_data.target_names
    feature_names = raw_data.feature_names
    # 设定根据y的值进行分层无偏采样
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                     shuffle=True,stratify=y,
                                                     random_state=0)
    return x_train,x_test,y_train,y_test,target_names,feature_names
#%% 使用DecisionTreeClassifier进行分类，返回graphviz图形
def do_DecisionTreeClassifier(*data):
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    clf = DecisionTreeClassifier()
    clf.fit(x_train,y_train)
    dot_data = export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)   
    print_DecisionTreeClassifier_info(clf,
                                          '使用默认参数构建的决策树分类器',
                                          y_true=y_test,
                                          y_pred=clf.predict(x_test),
                                          feature_names=feature_names,
                                          target_names=target_names)
    return clf,graph

def print_DecisionTreeClassifier_info(clf,title,y_true,y_pred,feature_names,target_names):
    # 输出特征的重要度信息
    print('='*35)
    print(title)
    print('各特征的重要性：')
    for feature_name,importance in zip(feature_names,clf.feature_importances_):
        print(feature_name,'\t=',importance)
    print('')
    print('max_features_ =',clf.max_features_)
    print('n_classes_ = ',clf.n_classes_)
    print('n_features_ =',clf.n_features_)
    print('n_outputs_ =',clf.n_outputs_)
    
    # 输出分类器的在测试集上的性能报告    
    print('分类器的在测试集上的性能报告')
    print(classification_report(y_true, y_pred, target_names=target_names))
    return
#%% 考察不同切分准则对分类性能的影响
def compare_DecisionTreeClassifier_criterion(*data):
    begin_t = datetime.datetime.now()
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    criterions = ['gini','entropy']
    clfs = []
    graphs = []
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(x_train,y_train)
        print_DecisionTreeClassifier_info(clf,
                                          'criterion = %s'%(criterion),
                                          y_true=y_test,
                                          y_pred=clf.predict(x_test),
                                          feature_names=feature_names,
                                          target_names=target_names)
        dot_data = export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
        graph = graphviz.Source(dot_data)
        clfs.append(clf)
        graphs.append(graph)
    end_t = datetime.datetime.now()
    print('耗时：',end_t - begin_t)
    return  clfs,graphs  
#%% 考察随机划分与最优划分对分类效果的影响
def compare_DecisionTreeClassifier_splitter(*data):
    begin_t = datetime.datetime.now()
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    splitters = ['best','random']
    clfs = []
    graphs = []
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(x_train,y_train)
        print_DecisionTreeClassifier_info(clf,
                                          'splitter = %s'%(splitter),
                                          y_true=y_test,
                                          y_pred=clf.predict(x_test),
                                          feature_names=feature_names,
                                          target_names=target_names)
        dot_data = export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
        graph = graphviz.Source(dot_data)
        clfs.append(clf)
        graphs.append(graph)
    end_t = datetime.datetime.now()
    print('耗时：',end_t - begin_t)
    return  clfs,graphs     
 
    
#%% 考察tree_depth对分类效果的影响
def compare_DecisionTreeClassifier_treeDepth(*data,maxdepth):
    begin_t = datetime.datetime.now()
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    max_depths = np.arange(1,maxdepth)
    clfs = []
    graphs = []
    train_scores = []
    test_scores =[]
    for maxdepth in max_depths:
        clf = DecisionTreeClassifier(max_depth=maxdepth)
        clf.fit(x_train,y_train)
        # 保持训练集与测试集的预测准确率
        train_scores.append(clf.score(x_train,y_train))
        test_scores.append(clf.score(x_test,y_test))
        # 输出分类器信息
#        print_DecisionTreeClassifier_info(clf,
#                                          'max_depth = %s'%(maxdepth),
#                                          y_true=y_test,
#                                          y_pred=clf.predict(x_test),
#                                          feature_names=feature_names,
#                                          target_names=target_names)
        # 生成决策树图
        dot_data = export_graphviz(clf, out_file=None,
                                    feature_names=feature_names,
                                    class_names=target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
        graph = graphviz.Source(dot_data)
        clfs.append(clf)
        graphs.append(graph)
    end_t = datetime.datetime.now()
    print('耗时：',end_t - begin_t)
    draw_compare_DecisionTreeClassifier_treeDepth(max_depths,train_scores,test_scores)
    return  clfs,graphs  

# 绘制不同深度的决策树的分类效果图   
def draw_compare_DecisionTreeClassifier_treeDepth(depths,train_scores,test_scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(depths,train_scores,'r-',label='训练集准确率')
    ax.plot(depths,test_scores,'b-',label='测试集准确率')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('准确率',fontproperties=myfont)
    ax.legend(loc='best',prop=myfont)
    plt.show()
    return
#%%
if __name__ == '__main__':
    x_train,x_test,y_train,y_test,target_names,feature_names = load_data()
    clf,graph = do_DecisionTreeClassifier(x_train,x_test,y_train,y_test,target_names,feature_names)
    # 打开视图
    #graph.view()
    
    # 分别考察不同参数设定对分类效果的影响
    clfs_criterion,graphs_criterion = compare_DecisionTreeClassifier_criterion(x_train,
                                                                               x_test,
                                                                               y_train,
                                                                               y_test,
                                                                               target_names,
                                                                               feature_names)
    
    clfs_splitter,graphs_splitter = compare_DecisionTreeClassifier_splitter(x_train,
                                                                               x_test,
                                                                               y_train,
                                                                               y_test,
                                                                               target_names,
                                                                               feature_names)
    
    clfs_treeDepth,graphs_treeDepth = compare_DecisionTreeClassifier_treeDepth(x_train,
                                                                               x_test,
                                                                               y_train,
                                                                               y_test,
                                                                               target_names,
                                                                               feature_names,
                                                                               maxdepth=100)
