import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc',size=10)

def runplt():
    plt.figure()
    plt.title('匹萨价格与直径数据',fontproperties=font)
    plt.xlabel('直径（英寸）',fontproperties=font)
    plt.ylabel('价格（美元）',fontproperties=font)
    # plt.axes([0,25,0,30])
    plt.grid(True)
    return plt

'''
ref: 2-linear-regression.pdf
多项式回归，一种特殊的多元线性回归方法，
增加了指数项（x的次数大于1）。
'''
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
# ==================================================
regressor = LinearRegression()
regressor.fit(X_train,y_train)
xx = np.linspace(0,26,100)
yy = regressor.predict(xx.reshape(xx.shape[0],1))
plt = runplt()
# 点图，和线性回归
plt.plot(X_train,y_train,'k.')
plt.plot(xx,yy)

# ===============================================
# 多项式回归,二元线性回归
quad_featurizer = PolynomialFeatures(degree=2)
# 将X_train转换为【1，x，x**2】的形式
X_train_quad = quad_featurizer.fit_transform(X_train)
X_test_quad = quad_featurizer.fit_transform(X_test)

regressor_quad = LinearRegression()
regressor_quad.fit(X_train_quad,y_train)
# np.array,将xx由一维数组，转化为【100*1】的数组
xx_quad = quad_featurizer.transform(xx.reshape(xx.shape[0],1))
plt.plot(xx,regressor_quad.predict(xx_quad),'r-')


print(X_train)
print(X_train_quad)
print(X_test)
print(X_test_quad)
print('一元线性回归 R-squared',regressor.score(X_test,y_test))
print('二元线性回归 R-squared',regressor_quad.score(X_test_quad,y_test))
# ================================================
# 多项式回归,三元线性回归
cubic_featurizer = PolynomialFeatures(degree=3)
X_train_cubic = cubic_featurizer.fit_transform(X_train)
X_test_cubic = cubic_featurizer.fit_transform(X_test)
regressor_cubic = LinearRegression()
regressor_cubic.fit(X_train_cubic,y_train)
# 数据变换维度
xx_cubic = cubic_featurizer.transform(xx.reshape(xx.shape[0],1))
plt.plot(xx,regressor_cubic.predict(xx_cubic),'g--')

plt.show()
print(X_test)
print(X_test_cubic)
print('三元线性回归 R-squared',regressor_cubic.score(X_test_cubic,y_test))