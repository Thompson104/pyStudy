'''
回归
'''
# %% 引用相关模块与函数
import numpy as np
import seaborn as sbn
import  matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

# %% 曲线定义函数
def f(x):
    return np.sin(x) + 0.5 * x

#%% 生成数据
## 未排序数据
#x = np.random.rand(50) * 4 * np.pi - 2 * np.pi
np.random.seed(0)
x = np.linspace(-2 * np.pi, 2 * np.pi,100) 
y= f(x)  + 2 * np.random.standard_normal(len(x))
plt.plot(x,y,'k.',label='rawdata')
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=42)
#%% 简单线性模型
plt.figure()

model = LinearRegression()
model.fit(train_x.reshape(-1, 1),train_y)
test_yy = model.predict(test_x.reshape(-1, 1))
plt.plot(train_x,train_y,'r*',label='训练数据')
plt.plot(test_x,test_yy,'y-',label='拟合数据')
plt.plot(test_x,test_y,'m.',label='测试数据')
plt.legend(loc='best',prop=myfont)
plt.show()
#%%
data = np.concatenate( ( x.reshape(-1,1),
                         (x**2).reshape(-1,1),
                         (x**3).reshape(-1,1),
                         np.sin(x).reshape(-1,1) )
                       ,axis=1 )
# 观察数据    
plt.plot(x,y,'k.',label='rawdata')
#%%拟合，自由度为freedeg
freedeg = 15
reg = np.polyfit(x,f(x),deg=freedeg)
ry = np.polyval(reg,x)
plt.plot(x,ry,'r.',label='regression')
# 生成多项式对象
pl = np.poly1d(reg)
print(reg)
print(pl) # 打印多项式
plt.show()

#%% sklearn中的线性模型

# 是scipy.linalg.lstsq的包装
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# 选择更好的基函数，可以取得更好的效果
model.fit(data,y)
yy = model.predict( data )
print('LinearRegression均方误差=',np.mean(yy - y))
print("LinearRegression的score=",model.score(data,y))
plt.plot(x,yy,'y*',label='LinearRegression')
#%% 岭回归
from sklearn.linear_model import Ridge
ridge_model = Ridge()
ridge_model.fit(data,y)
ridge_yy = ridge_model.predict( data)
print('Ridge均方误差=',np.mean(ridge_yy - y))
print("Ridge的score=",ridge_model.score(data,y))
plt.plot(x,ridge_yy,'g1',label='Ridge')

from sklearn.linear_model import Lasso
lasso_model = Lasso()
lasso_model.fit(data,y)
lasso_yy = lasso_model.predict( data)
print('Lasso均方误差=',np.mean(lasso_yy - y))
print("Lasso的score=",lasso_model.score(data,y))
plt.plot(x,lasso_yy,'c2',label='Lasso')

from sklearn.linear_model import ElasticNet
elasticNet_model = ElasticNet()
elasticNet_model.fit(data,y)
elasticNet_yy = elasticNet_model.predict( data)
print('ElasticNet均方误差=',np.mean(elasticNet_yy - y))
print("ElasticNet的score=",elasticNet_model.score(data,y))
plt.plot(x,elasticNet_yy,'m3',label='ElasticNet回归')


#%% 设置图形
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(loc='best',prop=myfont)
# 显示图例
plt.show()

## 简单的检查,freedeg = 15
#print(np.allclose(f(x),ry))
#
#print('均方差=', np.sum( ( f(x)-ry )**2 ) / len(x) )
#
#plt.figure()
#matrix = np.zeros((3+1,len(x)))
#matrix[3,:] = x ** 3
#matrix[3,:] = np.sin(x) # 用sin函数代替x^3
#matrix[2,:] = x ** 2
#matrix[1,:] = x ** 1
#matrix[0,:] = x ** 0
## 估计线性模型中的系数：a=np.linalg.lstsq(x,b),有b=a*x
#reg = np.linalg.lstsq(matrix.T,f(x))[0]
#print(reg)
#ry = np.dot(reg,matrix)
#
#plt.title('估计线性模型中的系数：a=np.linalg.lstsq(x,b),有b=a*x')
#plt.plot(x,ry,'r.',label='regression')
#plt.plot(x,f(x),'b',label='f(x)')