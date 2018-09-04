# -*- coding: utf-8 -*-
'''

'''

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression
import sklearn.metrics as ms



x = np.array([1.5,2.8,4.5,7.5,10.5,13.5,15.1,16.5,19.5,22.5,24.5,26.5])
x = x.reshape(-1,1)
y = np.array([7.0,5.5,4.6,3.6,2.9,2.7,2.5,2.4,2.2,2.1,1.9,1.8])

#plt.plot(x,y,'ro')

# 一般线性回归
model = LinearRegression()
model.fit(x,y)
print('一般线性回归R方值',model.score(x,y))
print('一般线性回归系数=',model.coef_)
print('一般线性回归截距=',model.intercept_)
# 多项式多项式
x_square = np.hstack((x**2,x**3,x**4,x**5))
model.fit(x_square,y)
print('多项式线性回归R方值',model.score(x_square,y))
print('多项式线性回归系数=',model.coef_)
print('多项式线性回归截距=',model.intercept_)

# 对数法变换多项式
x_log = np.log(x)
model.fit(x_log,y)
print('Log变换多项式线性回归R方值',model.score(x_log,y))
print('Log变换多项式线性回归系数=',model.coef_)
print('Log变换多项式线性回归截距=',model.intercept_)

# 指数变换多项式
y_log = np.log(y)
model.fit(x,y_log)
print('指数多项式线性回归R方值',model.score(x,y_log))
print('指数多项式线性回归系数=',model.coef_)
print('指数多项式线性回归截距=',model.intercept_)

# 幂函数多项式
x_log = np.log(x)
y_log = np.log(y)
model.fit(x_log,y_log)
print('幂函数多项式线性回归R方值',model.score(x_log,y_log))
print('幂函数多项式线性回归系数=',model.coef_)
print('幂函数多项式线性回归截距=',model.intercept_)