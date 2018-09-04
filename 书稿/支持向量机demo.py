# -*- coding: utf-8 -*-
import numpy as np #科学计算
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

from sklearn import datasets as ds
from sklearn import linear_model as lm
from sklearn import svm
from sklearn import model_selection as ms
from sklearn import metrics


#%% 数据加载
def load_data():
    raw_data = np.loadtxt(fname=r'..\input\wine.data',delimiter=',')
    X = raw_data[:,1:]
    y = raw_data[:,0]
    x_train,x_test,y_train,y_test = ms.train_test_split(X,y,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        shuffle=True)
    return x_train,x_test,y_train,y_test

def create_data(n=100):
    '''
    生成线性可分数据库
    '''
    # np.random.seed()的作用：使得随机数据可预测。
    # 当我们设置相同的seed，每次生成的随机数相同。
    # 如果不设置seed，则每次会生成不同的随机数
    np.random.seed(0)
    # numpy.random.randint(low, high=None, size=None, dtype=’l’)
    # 返回随机整数，范围区间为[low,high），包含low，不包含high
    # size为数组维度大小
    size = (n,1)
    x_11 = np.random.randint(0,100,size)
    x_12 = np.random.randint(0,100,size)
    x_13 = 20 + np.random.randint(0,10,size)
    
    x_21 = np.random.randint(0,100,size)
    x_22 = np.random.randint(0,100,size)
    x_23 = 10 - np.random.randint(0,10,size)
    
    # 沿X轴旋转45度，y = 0.5 * 2**.5 *  ( x1 - x2 )
    new_x_12 = (np.sqrt(2) / 2) * (x_12 - x_13)
    new_x_13 = (np.sqrt(2) / 2) * (x_12 + x_13) 
    new_x_22 = (np.sqrt(2) / 2) * (x_22 - x_23) 
    new_x_23 = (np.sqrt(2) / 2) * (x_22 + x_23)
    
    #
    plus_samples = np.hstack([x_11,new_x_12,new_x_13,np.ones(size)])
    minus_sample = np.hstack([x_21,new_x_22,new_x_23,-np.ones(size)])
    samples = np.vstack([plus_samples,minus_sample])
    x_train,x_test,y_train,y_test = ms.train_test_split(samples[:,0:-1],
                                                        samples[:,-1],
                                                        test_size=0.3,
                                                        random_state=0,
                                                        shuffle=True)
    return x_train,x_test,y_train,y_test
    
    return samples
#%% 线性分类svm
def LinearSVC(*data):
    clf = svm.LinearSVR()
    clf.fit(x_train,y_train)
    score = clf.score(x_test,y_test)
    print("各特征的权重:\n%s"%(clf.coef_))
    print('决策函数的截距',clf.intercept_)
    print(score)
    print('')    
    y_pred=clf.predict(x_test)
    print( metrics.classification_report( y_true=y_test,y_pred=y_pred ) )
    return
# 考虑不同损失函数
def LinearSVC_loss(*data):
    losses = ['epsilon_insensitive','squared_epsilon_insensitive']
    for loss in losses:        
        clf = svm.LinearSVR(loss=loss,random_state=0)
        clf.fit(x_train,y_train)
        score = clf.score(x_test,y_test)
        print('损失函数 = %s'%(loss))
        print("各特征的权重:\n%s"%(clf.coef_))
        print('决策函数的截距\n',clf.intercept_)
        print('正确率：',score)
        print('')
    return
    
#%% 主程序
if __name__ == '__main__':
#    x_train,x_test,y_train,y_test = load_data()
    x_train,x_test,y_train,y_test = create_data()
    LinearSVC(x_train,x_test,y_train,y_test)
#    LinearSVC_loss(x_train,x_test,y_train,y_test)
    