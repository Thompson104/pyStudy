print(__doc__)
'''
多元线性回归
'''
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
# 手工计算
# y = x * bata,求bata
x = [[1,6,2],[1,8,1],[1, 10, 0], [1, 14, 2], [1, 18, 0]]
y = [[7], [9], [13], [17.5], [18]]
bata = np.dot( np.linalg.inv( np.dot(np.transpose(x),x) ),
               np.dot(np.transpose(x),y) )
print(bata)
# 通过最小二乘法函数来求
bata1 = np.linalg.lstsq(x,y)

# 通过sklearn
x = [[6,2],[8,1],[10, 0], [14, 2], [18, 0]]
model = LinearRegression()
model.fit(x,y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X_test)
for i ,prediction in enumerate(X_test):
    print('Predicted: %s, Target: %s' %(prediction,y_test[i]))
print('R-squared: %.2f' % (model.score(X_test,y_test)))

