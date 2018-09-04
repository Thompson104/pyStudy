# -*- coding: utf-8 -*-
"""
交叉验证范例

@author: TIM
"""
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn import datasets as ds
from sklearn import svm

if __name__ == '__main__':
    iris = ds.load_iris()
    
    x_train,x_test,y_train,y_test = cv.train_test_split(
            iris.data,iris.target,test_size=0.4,random_state=0)
    # 创建模型
    clf = svm.SVC(kernel='rbf',C=1.0)
    # 训练模型
    #clf.fit(x_train,y_train)
    # 验证训练集
    #score = clf.score(x_test,y_test)
    
    ## 直接进行5折交叉验证
    
    # =============================================
    # 实现CV最简单的方法是cross_validation.cross_val_score函数
    # =============================================
    # 通过cross_validation.ShuffleSplit生成一个CV迭代策略生成器cv
    mycv = cv.ShuffleSplit(iris.data.shape[0],n_iter=10,test_size=0.3,random_state=1)
    scores1 = cv.cross_val_score(clf,iris.data,iris.target,cv=mycv)
    # 最基础的CV算法，也是默认采用的CV策略​。
    # 主要的参数包括两个，一个是样本数目，一个是k-fold要划分的份数
    kf = cv.KFold(iris.data.shape[0],n_folds=10,random_state=1)
    scores2 = cv.cross_val_score(clf,iris.data,iris.target,cv=kf)
    # 与k-fold类似，将数据集划分成k份，
    #不同点在于，划分的k份中，每一份内各个类别数据的比例和原始数据集中各个类别的比例相同
    sf = cv.StratifiedKFold(iris.target,n_folds=10,random_state=1)
    scores3 = cv.cross_val_score(clf,iris.data,iris.target,cv=sf)
    #  Leave-one-out
    leaveOneOut = cv.LeaveOneOut(iris.data.shape[0])
    scores4 = cv.cross_val_score(clf,iris.data,iris.target,cv=leaveOneOut)
    
    # Leave-P-out 每次从整体样本中去除p条样本作为测试集
    #如果共有n条样本数据，那么会生成c(n,p)个训练集/测试集对。
    # 和LOO，KFold不同，这种策略中p个样本中会有重叠。
    lpo = cv.LeavePOut(iris.data.shape[0],2) #一万一千多个
    scores5 = cv.cross_val_score(clf,iris.data,iris.target,cv=lpo)
    
    # Leave-one-label-out
    """
    这种策略划分样本时，会根据第三方提供的整数型样本类标号进行划分。
    每次划分数据集时，取出某个属于某个类标号的样本作为测试集，剩余的作为训练集。
    """
    
    # Leave-P-Label-Out
    """
    与Leave-One-Label-Out类似，但这种策略每次取p种类标号的数据作为测试集，
    其余作为训练集。
    """
    
