# -*- coding: utf-8 -*-
import numpy as np #科学计算
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.datasets as ds
#%% 数据模块
def get_data():
    raw_data = ds.load_iris()
    X = raw_data.data[:,0:2]
    y = raw_data.target
    feature_names = raw_data.feature_names
    classes_names = raw_data.target_names
    x_train,x_test,y_train,y_test = train_test_split(X,y,
                                                     random_state=0,
                                                     test_size=0.3,
                                                     shuffle = True, # 打乱样本
                                                     stratify=y # 保持类别比例
                                                     )
    return x_train,x_test,y_train,y_test,feature_names,classes_names

#%% 绘图单元
    
def plot_samples(ax,x,y,classes_names):
    n_classes = np.unique(y).shape[0]
    for i,name in zip( np.arange(n_classes),classes_names) :
        index = np.where(y==i)
        ax.scatter(x[index,0],x[index,1],
                   label=name,
                   cmap='ocean')
    return

def plot_classifier_predict_meshgrid(ax,clf,
                                     x_min,x_max,
                                     y_min,y_max):
    plot_step = 0.2
    (xx,yy) = np.meshgrid( np.arange(x_min,x_max,plot_step),
                         np.arange(y_min,y_max,plot_step) )
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx,yy,z,cmap='hot')
    return
'''
打印模型的参数
'''
def print_model(clf,title):
    print(title)
    print('权重：',clf.coefs_)
    return

#%% 分类
def mlpclassifier_iris(x_train,y_traint,x_test,y_test,
                       x_min,x_max,y_min,y_max,
                       classes_name,feature_names):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    clf = MLPClassifier(activation='logistic',max_iter=10000,hidden_layer_sizes=(30,))
    clf.fit(x_train,y_traint)
    train_score = clf.score(x_train,y_traint)
    test_score = clf.score(x_test,y_test)
    plot_classifier_predict_meshgrid(ax,clf,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)
    plot_samples(ax,x_train,y_train,classes_name)
    plt.legend(loc='best')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('训练准确率：%0.2f  测试准确率：%0.2f'%(train_score,test_score),fontproperties=myfont)
    fig.tight_layout()
    plt.show()
    return

#%% 考察不同的隐藏层对分类效果的影响
def mlpclassifier_iris_hidden_layer_sizes():
    max_iter = 10000
    fig = plt.figure(figsize=(15,9))
    hidden_layer_sizes = [(10,),(30,),(100,),(5,5),(10,10),(30,30)]
    for i ,size in enumerate(hidden_layer_sizes):
        ax = fig.add_subplot(2,3,i+1)
        clf = MLPClassifier(activation='logistic',
                            max_iter=max_iter,
                            hidden_layer_sizes=size)
        clf.fit(x_train,y_train)
        train_score = clf.score(x_train,y_train)
        test_score = clf.score(x_test,y_test)
        plot_classifier_predict_meshgrid(ax,clf,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)
        plot_samples(ax,x_train,y_train,classes_name)
        plt.legend(loc='best')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('训练准确率：%0.2f,测试准确率：%0.2f,隐藏层：%s'%(train_score,test_score,size),
                     fontproperties=myfont)
    fig.tight_layout()
    plt.show()
    return
#%% 考察不同的激活函数对分类效果的影响
def mlpclassifier_activations():
    max_iter = 10000
    size = (10,)
    fig = plt.figure(figsize=(9,6))
    activations = ['logistic','relu','tanh','identity']
    for i ,activation in enumerate(activations):
        ax = fig.add_subplot(1,len(activations),i+1)
        clf = MLPClassifier(activation=activation,
                            max_iter=max_iter,
                            hidden_layer_sizes=size)
        clf.fit(x_train,y_train)
        train_score = clf.score(x_train,y_train)
        test_score = clf.score(x_test,y_test)
        plot_classifier_predict_meshgrid(ax,clf,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)
        plot_samples(ax,x_train,y_train,classes_name)
        plt.legend(loc='best')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('训练准确率：%0.2f \n测试准确率：%0.2f\n迭代次数：%s \n激活函数：%s'
                     %(train_score,test_score,clf.n_iter_,activation),
                     fontproperties=myfont)
    fig.tight_layout(pad=0.4, w_pad=0.0, h_pad=1.0)
    plt.show()
    return

#%% 考察不同的优化算法对分类效果的影响
def mlpclassifier_algorithms():
    max_iter = 10000
    size = (10,)
    fig = plt.figure(figsize=(9,6))
    algorithms = ['sgd','adam','lbfgs']
    for i ,algorithm in enumerate(algorithms):
        ax = fig.add_subplot(1, len(algorithms) ,i+1)
        clf = MLPClassifier(activation='relu',
                            solver=algorithm,
                            max_iter=max_iter,
                            hidden_layer_sizes=size)
        clf.fit(x_train,y_train)
        train_score = clf.score(x_train,y_train)
        test_score = clf.score(x_test,y_test)
        plot_classifier_predict_meshgrid(ax,clf,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)
        plot_samples(ax,x_train,y_train,classes_name)
        plt.legend(loc='best')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('训练准确率：%0.2f \n测试准确率：%0.2f \n算法：%s'%(train_score,test_score,algorithm),
                     fontproperties=myfont)
    fig.tight_layout(pad=0.4, w_pad=0.0, h_pad=1.0)
    plt.show()
    return   

#%% 考察不同的学习率对分类效果的影响
def mlpclassifier_etas():
    max_iter = 10000
    size = (10,)
    fig = plt.figure(figsize=(9,6))
    etas = np.logspace(-3,0,4)
    for i ,eta in enumerate(etas):
        ax = fig.add_subplot(1, len(etas) ,i+1)
        clf = MLPClassifier(activation='relu',
                            solver='adam',
                            learning_rate_init=eta,
                            learning_rate='constant',
                            max_iter=max_iter,
                            hidden_layer_sizes=size)
        clf.fit(x_train,y_train)
        train_score = clf.score(x_train,y_train)
        test_score = clf.score(x_test,y_test)
        plot_classifier_predict_meshgrid(ax,clf,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)
        plot_samples(ax,x_train,y_train,classes_name)
        plt.legend(loc='best')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('训练准确率：%0.2f \n测试准确率：%0.2f \n学习率：%s'%(train_score,test_score,eta),
                     fontproperties=myfont)
    fig.tight_layout(pad=0.4, w_pad=0.0, h_pad=1.0)
    plt.show()
    return  
#%% 考察不同的学习率更新策略对分类效果的影响
def mlpclassifier_learning_rates():
    max_iter = 10000
    size = (10,)
    fig = plt.figure(figsize=(9,6))
    learning_rates = ['constant', 'invscaling', 'adaptive']
    for i ,learning_rate in enumerate(learning_rates):
        ax = fig.add_subplot(1, len(learning_rates) ,i+1)
        clf = MLPClassifier(activation='relu',
                            solver='sgd',
                            learning_rate_init=.001,
                            learning_rate=learning_rate,
                            power_t = 0.8,
                            max_iter=max_iter,
                            hidden_layer_sizes=size,
                            verbose=True)
        clf.fit(x_train,y_train)
        train_score = clf.score(x_train,y_train)
        test_score = clf.score(x_test,y_test)
        plot_classifier_predict_meshgrid(ax,clf,x_min=x_min,x_max=x_max,y_min=y_min,y_max=y_max)
        plot_samples(ax,x_train,y_train,classes_name)
        plt.legend(loc='best')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title('训练准确率：%0.2f \n测试准确率：%0.2f \n学习更新策略：%s'
                     %(train_score,test_score,learning_rate),
                     fontproperties=myfont)
        print_model(clf,'考察不同的学习率更新策略对分类效果的影响')
    fig.tight_layout(pad=0.4, w_pad=0.0, h_pad=1.0)
    plt.show()
    return  
#%% 主程序
if __name__ == '__main__': 
    print("神经网络代码范例")
    x_train, x_test, y_train, y_test, feature_names, classes_name = get_data()
    x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 2
    y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_samples(ax,
                 np.vstack((x_train,x_test)),
                 np.vstack((y_train[:,None],y_test[:,None])),
                 classes_name)
    mlpclassifier_iris(x_train,y_train,x_test,y_test,
                       x_min,x_max,y_min,y_max,
                       classes_name,feature_names)
    mlpclassifier_iris_hidden_layer_sizes()
    mlpclassifier_activations()
    mlpclassifier_algorithms()
    mlpclassifier_etas()
    mlpclassifier_learning_rates()
    
        
            