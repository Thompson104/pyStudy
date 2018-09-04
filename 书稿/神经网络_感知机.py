# -*- coding: utf-8 -*-
import numpy as np #科学计算
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

from sklearn.neural_network import MLPClassifier
import sklearn.datasets as ds
#%% 数据模块
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
    np.random.shuffle(samples)
    return samples


#%% 绘制样本数据的三维图
def plot_sample(ax,samples):
    y = samples[:,-1]
    X = samples[:,0:-1]
    ax.scatter(X[y==1,0],X[y==1,1],X[y==1,2],marker='+',color='b')
    ax.scatter(X[y==-1,0],X[y==-1,1],X[y==-1,2],marker='^',color='y')
    return
#%% 感知机
def perceptron(samples,w_0,b_0,eta=0.1):
    train_x = samples[:,0:-1]
    train_y = samples[:,-1]
    length = train_x.shape[0]
    w = w_0
    b = b_0
    step_num = 0
    while True:
        i = 0
        while (i < length):
            step_num +=1
            if step_num > 400000:
                print('训练失败，终止于第{num}次循环'.format(num=step_num-1))
                return None
#            print('step_num = ',step_num)
            x_i = train_x[i].reshape((-1,1))
            y_i = train_y[i]
            if y_i * ( np.dot(w.T,x_i) + b ) <= 0:
                w = w + eta * y_i * x_i
                b = b + eta * y_i 
#                print('break')
                break
            else:
                i = i + 1
        if( i == length):
            break
    return ( w,b,step_num )
 
def create_hyperplane(x,y,w,b):
    '''
    生成超平面
    '''
    return (  -w[0][0] * x - w[1][0] * y - b ) / w[2][0]
#%% 感知机的对偶形式
def create_w(train_data,alpha):
    x = train_data[:,0:-1]
    y = train_data[:,-1]
    N = train_data.shape[0]
    w = np.zeros((x.shape[1],1))
    for i in np.arange(0,N):
        w = w + alpha[i][0] * y[i] * (x[i].reshape((-1,1)))    
    return w

def perceptron_dual(train_data,eta,alpha_0,b_0):
    '''
    感知机的对偶形式
    '''
    x = train_data[:,0:-1]
    y = train_data[:,-1]
    length = train_data.shape[0]
    alpha = alpha_0
    b=b_0
    step_num =0
    while True:
        i = 0
        while ( i < length):
            step_num +=1
            x_i = x[i].reshape((-1,1))
            y_i = y[i]
            w = create_w(train_data,alpha)
#            print('w = ',w)
            z = y_i * ( np.dot( w.T,x_i ) + b )
            if z <=0:
                alpha[i][0] += eta
                b += eta * y_i
                break
            else:
                i += 1
                pass
            pass
        if(i == length):
            break
        pass
    return (alpha,b,step_num)
    
    return
    
#%% 
if __name__ == '__main__': 
    print("神经网络代码范例")
    control = 2
    if control == 1:        
        samples = create_data()
        # 参数设定
        eta,w_0,b_0 = 0.1,np.ones((3,1),dtype=float),1
        result = perceptron(samples,w_0,b_0,eta)
        if result == None :
            import sys
            sys.exit(-1)
        else:
            w,b,num = result
        print('w = ',w,' b = ',b,'num = ',num)
        
        fig = plt.figure()
        plt.suptitle('单层感知机',fontproperties=myfont)
        ax = Axes3D(fig)
        # 绘制样本点
        plot_sample(ax,samples)
        
        # 绘制分离超平面
        x = np.linspace(-30,100,100)
        y = np.linspace(-30,100,100)
        x,y = np.meshgrid(x,y)
        z = create_hyperplane(x,y,w,b)
#        ax.plot_surface(x,y,z,color='g')    
        ax.legend(loc='best')
        plt.show()
        pass
    elif control == 2:
        samples = create_data()
        # 参数设定
        eta,w_0,b_0 = 0.1,np.ones((3,1),dtype=float),1        
        w_1,b_1,num_1 = perceptron(samples,w_0,b_0,eta)
        alpha,b_2,num_2 = perceptron_dual(samples,eta,
                                  alpha_0=np.zeros((samples.shape[0]*2,1)),
                                  b_0=0
                                  )  
        w_2 = create_w(samples,alpha)
        print('权重 = ',w_1,' 截距 = ',b_1,'迭代次数 = ',num_1)
        print('alpha权重 = ',alpha,
              ' 截距 = ',b_2,
              '迭代次数 = ',num_2,
              '权重 = ',w_2)
        
        fig = plt.figure()
        plt.suptitle('单层感知机',fontproperties=myfont)
        ax = Axes3D(fig)
        # 绘制样本点
        plot_sample(ax,samples)
        
        # 绘制分离超平面
        x = np.linspace(-30,100,100)
        y = np.linspace(-30,100,100)
        x,y = np.meshgrid(x,y)
        z_1 = create_hyperplane(x,y,w_1,b_1)
        z_2 = create_hyperplane(x,y,w_2,b_2)
        ax.plot_surface(x,y,z_1,color='r',alpha=0.8,cstride=1) 
#        ax.plot_surface(x,y,z_2,color='c',alpha=0.8,cstride=1) 
        ax.legend(loc='best')
        plt.show()
        
            