# -*- coding: utf-8 -*-
"""
一个简单的神经网络
"""
import numpy as np
import matplotlib.pyplot as plt
'''
最初研究人口增长
'''
def logistic(x):
    y = 1/(1 + np.exp(-x))
    return y
'''
logistic函数的导数
'''
def logistic_deriv(x):
    return logistic(x) * (1.0 - logistic(x))

def tanh(x):
    y = ( np.exp(x) -np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )
    #y = np.tanh(x)
    return y
'''
tanh函数的导数
'''
def tanh_deriv(x):
    y = 1.0 - tanh(x) **2
    return y

class NeuralNetwork(object):
    def __init__(self,layers,activation='tanh'):
        '''
        '''
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        
        self.weights = []
        for i in np.arange(1,len(layers)):
            '''
            第i层的单元与前层单元之间的权重,与前层单元多一个bias
            '''
            # 对每一层的权重都要初始化初始值范围在-0.25~0.25之间，然后保存在weight中
            self.weights.append( ( 2 * np.random.random( (layers[i-1] + 1 , layers[i]   + 1 ) ) - 1 ) * 0.25 )
        #print(self.weights)
        return
    
    def fit(self,X,y,learning_rate=0.2,epochs=1000):
        # =====================================================
        # 对数据进行简单处理
        # =====================================================
        
        X = np.atleast_2d(X)#判断输入训练集是否为二维
        temp = np.ones([X.shape[0],X.shape[1]+1])#列加1是因为最后一列要存入标签分类，这里标签都为1
        temp[:,0:-1] = X
        X = temp
        y = np.array(y)#训练真实值
        
        # =====================================================
        for k in range(epochs):#循环
            i = np.random.randint(X.shape[0])#随机选取训练集中的一个
            a = [X[i]]
            #计算激活值
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))

            error = y[i] - a[-1]#计算偏差
            deltas = [error*self.activation_deriv(a[-1])]#输出层误差
            #下面计算隐藏层     
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            #下面开始更新权重和偏向
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
            
        
        return
        #预测函数
    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
        
    
nn = NeuralNetwork([2,5,5,8,5,1],'tanh')
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
nn.fit(X, y)
for i in [[0,0],[0,1],[1,0],[1,1]]:
    print(i, nn.predict(i))


