# -*- coding: utf-8 -*-
import numpy as np #科学计算
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)


def sigmold(x):
    y = 1/( 1 + np.exp(-x))
    return y

def anti_sigmold(x):
    y = sigmold(x) * ( 1 - sigmold(x) )
    return y

def tanh(x):
    y = ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )
    return y
def anti_tanh(x):
    y =  1 - tanh(x)**2
    return y

def ReLU(x):
    return abs(x) * (x > 0) 
def anti_ReLU(x):
    y = 1 * (x > 0)
    return y

def softplus(x):
    y = np.log( 1 + np.exp(x) )    
    return y
def anti_softplus(x):
    y = 1 / ( 1 + np.exp(-x) )
    return y
'''
阶跃函数
'''
def f(T):
    def wrap(t):
        if t > 0 and t < T / 2: return 1
        elif t == T / 2: return 0
        else:return -1
#%% 
if __name__ == '__main__':
    x = np.linspace(-5,5,50)   
    figsize = (9,6)
#%%      
    fig1 = plt.figure(figsize=figsize)
    y = sigmold(x)
    ax = plt.subplot(121)
    plt.plot(x,y,'r-')
    ax.text(7,10,'test')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('sigmold')
    
    ax = plt.subplot(122)
    y = anti_sigmold(x)
    print(x)
    print(y)
    plt.plot(x,y,'r-')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('sigmold的导数',fontproperties=myfont)
    plt.show()
#%%  
    fig2 = plt.figure(figsize=figsize)
    y = tanh(x)
    ax =plt.subplot(121)
    plt.plot(x,y,'r-')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('tanh')
    
    y = anti_tanh(x)
    ax =plt.subplot(122)
    plt.plot(x,y,'r-')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('tanh的导数',fontproperties=myfont)   
    
    plt.show()
#%%    
    fig3 = plt.figure(figsize=figsize)
    y = ReLU(x)
    ax =plt.subplot(121)
    plt.plot(x,y,'r-')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('ReLU')
    
    y = anti_ReLU(x)
    ax =plt.subplot(122)
    plt.plot(x,y,'r-')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('ReLU的导数',fontproperties=myfont)   
    
    plt.show()
#%%      
    fig4 = plt.figure(figsize=figsize)
    y = softplus(x)
    ax =plt.subplot(121)
    plt.plot(x,y,'r-')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('softplus')
    
    y = anti_softplus(x)
    ax =plt.subplot(122)
    plt.plot(x,y,'r-')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_title('softplus的导数',fontproperties=myfont)   
    
    plt.show()    
#%%
    fig5 = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plus_x = x[x>=0]
    minu_x = x[x<0]
    plt.plot(plus_x,np.ones((plus_x.shape)),'k-',
             minu_x,np.ones((minu_x.shape)) * -1,'k-')
    ax.spines['top'].set_color('none')  
    ax.spines['right'].set_color('none') 
    ax.xaxis.set_ticks_position('bottom')  
    ax.spines['bottom'].set_position(('data',0)) 
    ax.spines['left'].set_position(('data',0))
    ax.set_title('阶跃函数',fontproperties=myfont)
    plt.show()