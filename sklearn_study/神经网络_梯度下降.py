# -*- coding: utf-8 -*-
"""
神经网络简单实现
"""
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    f = gzip.open('../input/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class Network(object):
    def __init__(self,sizes,sigmoid='logistic',eta=0.01):
        '''
        sizes:list,每层单元个数
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 初始化隐藏层,输出层的bias
        self.biases = np.array( [np.random.randn(y,1) for y in sizes[1:]])
        # 初始化输入层,隐藏层的weights,每i行=后一层i单元与前一层单元的连接权重
        self.weights = np.array( [np.random.randn(y,x)
                                  for y,x in zip(sizes[1:], sizes[0:-1])] )
        # 初始化隐藏层,输出层等个单元的值
        self.values = np.array( [ np.ones((y,1)) for y in sizes[1:]  ]  )
        #
        if sigmoid == 'logistic':
            self.sigmoid = self.logistic
            self.sigmoid_deriv = self.logistic_deriv
        elif sigmoid == 'tanh':
            self.sigmoid = self.tanh
            self.sigmoid_deriv = self.tanh_deriv
        else:
            self.sigmoid = self.logistic
            self.sigmoid_deriv = self.logistic_deriv
        return
    '''
    前向计算输出
    '''
    def feedforword(self,inputvalues):
#        outputvalues=np.array(inputvalues)
        for b,w in zip(self.biases,self.weights):
            inputvalues = self.sigmoid(np.dot(w,inputvalues)) + b
        return inputvalues
    '''
    反向误差传播,返回更新后的权重和biases
    '''
    def backprop(self,x,y):
        # ▽读作nabla,初始化biase和weight的偏导矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # =====================================================================
        # 1, 前向计算
        activation = x.reshape((x.shape[0],1))
        activations = [x] # 所有的activation,一层一层
        zs = []             #中间变量,sum w*a + b
        for b,w in zip(self.biases,self.weights):
            # 激活函数之前
            z = np.dot(w,activation) + b
            zs.append(z)
            # 激活函数之后,激活值
            activation = self.sigmoid(z)
            activations.append(activation)
        # =====================================================================
        # 2,反向传播误差,学习率不考虑即为1
        
        # 首先,输出层根据误差函数偏导修正其与相连隐藏层单元之间的权重
        # 然后,循环.上一层隐藏层根据公式修正与其相连的隐藏层或输入层单元之间的权重        
        
        # 2.1 计算输出层的error的偏导
        # delta Δ = g_j
        delta = self.cost_deriv( activations[-1] , y ) * self.sigmoid_deriv( zs[-1] )
        # 输出层的bias和权重的偏导
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot( delta,activations[-2].transpose() )
        
        # 2.2 计算隐藏层的b和w的偏导
        for n in np.arange(2,self.num_layers):
            z           = zs[-n]
            sp = self.sigmoid_deriv(z)
            # -i层的隐藏层对该层输出z的偏导
            delta = sp * np.dot( nabla_w[-n + 1].transpose(), delta )
            # 
            nabla_b[-n] = delta
            nabla_w[-n] = np.dot( delta,activations[-n - 1].transpose())
        return (nabla_b,nabla_w)
    '''
    计算
    '''
    def cost_deriv(self,outputs,y):
        '''返回累计误差的偏导'''
        return (outputs - y)
    
    def logistic(self,x):
        y = 1/(1 + np.exp(-x))
        return y
    
    '''
    logistic函数的导数
    '''
    def logistic_deriv(self,x):
        return self.logistic(x) * (1.0 - self.logistic(x))
    
    def tanh(self,x):
        y = ( np.exp(x) -np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )
        #y = np.tanh(x)
        return y
    '''
    tanh函数的导数
    '''
    def tanh_deriv(self,x):
        y = 1.0 - self.tanh(x) **2
        return y
    
    def evaluate(self,x,y):
        test_results = [ ( np.argmax( self.feedforword(x) ),y) for x,y in zip(x,y) ]
        return sum(int(x==y) for (x,y) in test_results)
    # ========================================================================
    # 简单神经网络,单个样本进行学习
    # ========================================================================
    def simple_NN_fit(self,xx,yy,epochs,eta):
        for i in np.arange(epochs):
            for x,y in zip(xx,yy):
                delta_nabla_b, delta_nabla_w = self.backprop(x,y)
                self.weights = self.weights - eta * delta_nabla_w
                self.biases = self.biases -eta * delta_nabla_b               
            
        return
    
if __name__ == '__main__':
    np.random.seed(0)
    network = Network([784,15,15,10],sigmoid='tanh')
    x = np.load('..\input\MNIST_data.npy')
    y = np.load('..\input\MNIST_label.npy')
    y = y.reshape((70000,1))
    raw_data = np.hstack((x,y))
    np.random.shuffle(raw_data)
    network.simple_NN_fit(raw_data[:5,:-1],raw_data[:5,-1],epochs=10,eta=0.3)
    
    
