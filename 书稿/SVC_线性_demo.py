# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import datasets  as ds
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import classification_report
#%% 加载数据
def load_classification_data(databaseName):
    target_names = []
    feature_names = []
    if databaseName == 'iris':        
        raw_data = ds.load_iris()
        target_names = raw_data.target_names
        feature_names = raw_data.feature_names
        
    elif databaseName == 'digits':
        raw_data = ds.load_digits()
        target_names = np.array(raw_data.target_names).astype('str')        
        feature_names = None
        
    elif databaseName == 'breast_cance':
        raw_data = ds.load_breast_cancer()
        target_names = raw_data.target_names
        feature_names = raw_data.feature_names
        
    elif databaseName == 'wine':
        raw_data = ds.load_wine()
        target_names = raw_data.target_names
        feature_names = raw_data.feature_names
        pass 
    
    X = raw_data.data
    y = raw_data.target
    
    x_train,x_test,y_train,y_test = train_test_split(X,y,
                                                     random_state=0,
                                                     test_size=0.3,
                                                     shuffle = True, # 打乱样本
                                                     stratify=y # 保持类别比例
                                                     )
    return x_train,x_test,y_train,y_test,target_names,feature_names    

#%% 测试linearSVC在不同的数据集上的表现
def test_linearSVC(dataBaseNames):    
    dataBaseNames = dataBaseNames
    scores = []
    for dataBaseName in dataBaseNames:
        print('数据集：',dataBaseName)
        clf = LinearSVC()
        x_train,x_test,y_train,y_test,target_names,feature_names = load_classification_data(dataBaseName)
        clf.fit(x_train,y_train)
        scores.append( cross_val_score(clf,x_test,y_test,cv=10)  )
        print(classification_report(y_test,
                                    clf.predict(x_test),
                                    target_names=target_names))
        pass
    return np.array(scores)
#%% 测试SVC在wine数据集上的表现
def test_SVC(datasetName):
    # 加载训练集与测试集等信息
    x_train,x_test,y_train,y_test,target_names,feature_names = load_classification_data(datasetName)
    y_train = y_train + 1
    y_test = y_test + 1
    # 构建分类器，并训练
    clf = SVC()    
    clf.fit(x_train,y_train)
    # 输出分类性能报告
    print(classification_report(y_test,
                                clf.predict(x_test),
                                target_names=target_names))
    # 进行10折交叉验证
    scores = cross_val_score(clf,x_test,y_test,cv=10)
    print("%15s : \tAccuracy= %0.2f, \tstd= %0.2f, \tmin=%.2f, \tmax=%.2f" 
          % (datasetName,scores.mean(), scores.std(),scores.min(),scores.max() ) )
    return    
#%%
if __name__ == '__main__':
#    dataBaseNames = ['iris','digits','breast_cance','wine']
#    scores =test_linearSVC(dataBaseNames)
#    for score,name in zip(scores,dataBaseNames):
#        print("%15s : \tAccuracy= %0.2f, \tstd= %0.2f, \tmin=%.2f, \tmax=%.2f" 
#              % (name,score.mean(), score.std(),score.min(),score.max() ) )
#        pass
    test_SVC('iris')
    
    