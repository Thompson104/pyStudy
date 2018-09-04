# -*- coding: utf-8 -*-
import numpy as np #科学计算
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

from sklearn.neural_network import MLPClassifier
import sklearn.datasets as ds
#%% 数据模块
def create_data_no_linear_2d(n=100):
    np.random.seed(0)
    size = (n,1)
    x_11 = np.random.randint(0,100,size)
    x_12 = 10 + np.random.randint(-5,5,size)
    x_21 = np.random.randint(0,100,size)
    x_22 = 20 + np.random.randint(0,10,size)
    x_31 = np.random.randint(0,100,(int(n/10),1))
    x_32 = 20 + np.random.randint(0,10,(int(n/10),1))
    
    # 沿X轴旋转45度，y = 0.5 * 2**.5 *  ( x1 - x2 )
    new_x_11 = (np.sqrt(2) / 2) * (x_11 - x_12)
    new_x_12 = (np.sqrt(2) / 2) * (x_11 + x_12) 
    new_x_21 = (np.sqrt(2) / 2) * (x_21 - x_22) 
    new_x_22 = (np.sqrt(2) / 2) * (x_21 + x_22)
    new_x_31 = (np.sqrt(2) / 2) * (x_31 - x_32) 
    new_x_32 = (np.sqrt(2) / 2) * (x_32 + x_32)
    #
    plus_samples = np.hstack([new_x_11,new_x_12,np.ones(size)])
    minus_samples = np.hstack([new_x_21,new_x_22,-np.ones(size)])
    err_samples   = np.hstack([new_x_31,new_x_32,np.ones((int(n/10),1))])
    
    samples = np.vstack([plus_samples,minus_samples,err_samples])
    np.random.shuffle(samples)
    return samples

def get_data():
    X,y = ds.load_iris(return_X_y=True)
    return

#%% 绘制样本数据的2维图
def plot_sample_2d(ax,samples):
    y = samples[:,-1]
    X = samples[:,0:-1]
    ax.scatter(X[y==1,0],X[y==1,1],marker='+',color='b')
    ax.scatter(X[y==-1,0],X[y==-1,1],marker='^',color='y')
    return
#%% 多层神经网络
def predict_with_MLPClassifier(ax,samples):
    train_x = samples[:,0:-1]
    train_y = samples[:,-1]
    clf = MLPClassifier(activation='logistic',max_iter=1000)
    clf.fit(train_x,train_y)
    print(clf.score(train_x,train_y))
    #
    x_min,x_max = train_x[:,0].min() - 1,train_x[:,0].max() + 2
    y_min,y_max = train_x[:,1].min() - 1,train_x[:,1].max() + 2
    
    plot_step = 1
    
    xx,yy = np.meshgrid(np.arange(x_min,x_max,plot_step),
                        np.arange(y_min,y_max,plot_step))
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx,yy,z,cmap='hot')
    
    return
    
#%% 
if __name__ == '__main__': 
    print("神经网络代码范例")     
    samples = create_data_no_linear_2d(100)
   
    fig = plt.figure()
    plt.suptitle('感知机')
    ax = fig.add_subplot(111)
    predict_with_MLPClassifier(ax,samples)
    # 绘制样本点
    plot_sample_2d(ax,samples)
    ax.legend(loc='best')
    plt.show()
        
            