# -*- coding: utf-8 -*-
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets  as ds
from  sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
#%% 数据加载单元
def load_data():
    raw_data = ds.load_wine()
    X = raw_data.data
    y = raw_data.target
    target_names = raw_data.target_names
    feature_names = raw_data.feature_names
    # 设定根据y的值进行分层无偏采样
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                     shuffle=True,stratify=y,
                                                     random_state=0)
    return x_train,x_test,y_train,y_test,target_names,feature_names

# 观察数据的分布情况
def plot_discover_data():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_train[:,6],x_train[:,12],marker='*',c=y_train)
    ax.set_xlabel(feature_names[6])
    ax.set_ylabel(feature_names[12])
    plt.show()
    return
#%% 高斯朴素贝叶斯分类器
def do_GaussianNB(*data):
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    clf = GaussianNB()
    clf.fit(x_train,y_train)
    print_GaussianNB_info(clf,'高斯朴素贝叶斯分类器参数')
    
    # 输出分类信息
    print('训练集：')
    print(classification_report(y_train,clf.predict(x_train)))
    print('测试集：')
    print(classification_report(y_test,clf.predict(x_test)))
    
    # 绘制混淆矩阵
    cnf_matrix = confusion_matrix(y_test,clf.predict(x_test))
    plot_confusion_matrix(cnf_matrix,classes=target_names,title='高斯朴素贝叶斯分类器的混淆矩阵')
    return

# 分类器信息输出
def print_GaussianNB_info(clf,title):
    print('='*35)
    print(title)
    print('class_prior_ size =',clf.class_prior_.shape,'\nclass_prior_ \n',clf.class_prior_)
    print('class_count_ size =',clf.class_count_.shape,'\nclass_count_ \n',clf.class_count_)
    print('theta_ size =',clf.theta_.shape,'\ntheta_ \n',clf.theta_)
    print('sigma_ size =',clf.sigma_.shape,'\nsigma_ \n',clf.sigma_)
    print()    
    return  
#%% 多项式朴素贝叶斯分类器
def do_MultinomialNB(*data):
    # 加载数据
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    # 训练分类器
    clf = MultinomialNB()
    clf.fit(x_train,y_train)
    
    # 输出分类器属性信息
    print_MultinomialNB_info(clf,'多项式朴素贝叶斯分类器参数')
    
    # 输出分类信息
    print('训练集：')
    print(classification_report(y_train,clf.predict(x_train)))
    print('测试集：')
    print(classification_report(y_test,clf.predict(x_test)))    
    
    # 绘制混淆矩阵
    cnf_matrix = confusion_matrix(y_test,clf.predict(x_test))
    plot_confusion_matrix(cnf_matrix,classes=target_names,title='多项式朴素贝叶斯分类器的混淆矩阵')
    return

# 分类器信息输出
def print_MultinomialNB_info(clf,title):
    print('='*35)
    print(title)
    print('class_log_prior_ size =',clf.class_log_prior_.shape,'\nclass_log_prior_ \n',clf.class_log_prior_)
    print('intercept_ size =',clf.intercept_.shape,'\nintercept_ \n',clf.intercept_)
    print('feature_log_prob_ size =',clf.feature_log_prob_.shape,'\nfeature_log_prob_ \n',clf.feature_log_prob_)
    print('coef_ size =',clf.coef_.shape,'\ncoef_',clf.coef_)
    print('class_count_ size =',clf.class_count_.shape,'\nclass_count_ \n',clf.class_count_)
    print('feature_count_ size =',clf.feature_count_.shape,'\nfeature_count_ \n',clf.feature_count_)
    
    print()    
    return  
#%% 伯努利朴素贝叶斯分类器
def do_BernoulliNB(*data):
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    clf = BernoulliNB()
    clf.fit(x_train,y_train)
    print_BernoulliNB_info(clf,'伯努利朴素贝叶斯分类器参数')
    print('训练集：')
    print(classification_report(y_train,clf.predict(x_train)))
    print('测试集：')
    print(classification_report(y_test,clf.predict(x_test)))
    
    # 绘制混淆矩阵
    cnf_matrix = confusion_matrix(y_test,clf.predict(x_test))
    plot_confusion_matrix(cnf_matrix,classes=target_names,title='伯努利朴素贝叶斯分类器的混淆矩阵')
    return

# 分类器信息输出
def print_BernoulliNB_info(clf,title):
    print('='*35)
    print(title)
    print('class_log_prior_ size =',clf.class_log_prior_.shape,'\nclass_log_prior_ \n',clf.class_log_prior_)
    print('feature_log_prob_ size =',clf.feature_log_prob_.shape,'\nfeature_log_prob_  \n',clf.feature_log_prob_)
    print('class_count_ size =',clf.class_count_.shape,'\nclass_count_ \n',clf.class_count_)
    print('feature_count_ size =',clf.feature_count_.shape,'\nfeature_count_ \n',clf.feature_count_)
    print()    
    return  
#%% 
def test_MultinomialNB_alpha(*data):
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    alphas = np.logspace(-2,5,200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        clf = MultinomialNB(alpha=alpha)
        clf.fit(x_train,y_train)
        train_scores.append(clf.score(x_train,y_train))
        test_scores.append(clf.score(x_test,y_test))
        pass
    # 绘图
    size = (9,6)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(alphas,train_scores,label='训练得分')
    ax.plot(alphas,test_scores,label='测试得分')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('得分',fontproperties=myfont)
    ax.set_xscale('log')
    ax.legend(loc='best',prop=myfont)
    plt.show()
    return    
#%% 
def test_BernoulliNB_alpha(*data):
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    alphas = np.logspace(-2,5,200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        clf = BernoulliNB(alpha=alpha)
        clf.fit(x_train,y_train)
        train_scores.append(clf.score(x_train,y_train))
        test_scores.append(clf.score(x_test,y_test))
        pass
    # 绘图
    size = (9,6)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(alphas,train_scores,label='训练得分')
    ax.plot(alphas,test_scores,label='测试得分')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('得分',fontproperties=myfont)
    ax.set_xscale('log')
    ax.legend(loc='best',prop=myfont)
    plt.show()
    return  
#%% 
def test_BernoulliNB_binarize(*data):
    x_train,x_test,y_train,y_test,target_names,feature_names = data
    x_min = np.min((np.min(x_train),np.min(x_test))) - 0.1
    x_max = np.max((np.max(x_train),np.max(x_test))) + 0.1
    binarizes = np.linspace(x_min,x_max)
    train_scores = []
    test_scores = []
    for binarize in binarizes:
        clf = BernoulliNB(binarize=binarize)
        clf.fit(x_train,y_train)
        train_scores.append(clf.score(x_train,y_train))
        test_scores.append(clf.score(x_test,y_test))
        pass
    # 绘图
    size = (9,6)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    ax.plot(binarizes,train_scores,label='训练得分')
    ax.plot(binarizes,test_scores,label='测试得分')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('得分',fontproperties=myfont)
    ax.set_xscale('log')
    ax.legend(loc='best',prop=myfont)
    plt.show()
    return 
#%% 绘制混淆矩阵
def plot_confusion_matrix(cm, classes,
                          title='混淆矩阵',
                          cmap=plt.cm.Blues):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontproperties=myfont)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('真实类别',fontproperties=myfont)
    plt.xlabel('预测类别',fontproperties=myfont)   
    plt.show()
    return
#%%
if __name__ == '__main__':
    x_train,x_test,y_train,y_test,target_names,feature_names = load_data()
    plot_discover_data()
    do_GaussianNB(x_train,x_test,y_train,y_test,target_names,feature_names)
    do_MultinomialNB(x_train,x_test,y_train,y_test,target_names,feature_names)
    do_BernoulliNB(x_train,x_test,y_train,y_test,target_names,feature_names)
    test_MultinomialNB_alpha(x_train,x_test,y_train,y_test,target_names,feature_names)
    test_BernoulliNB_alpha(x_train,x_test,y_train,y_test,target_names,feature_names)
    test_BernoulliNB_binarize(x_train,x_test,y_train,y_test,target_names,feature_names)