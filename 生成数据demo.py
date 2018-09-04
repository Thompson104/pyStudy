import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

'''
生产用于回归的数据
'''
from sklearn.datasets import make_regression
x,y,true_coefficient = make_regression(n_samples=200,
                                       n_features=4,
                                       n_informative=10,
                                       noise=100,
                                       coef=True,
                                       random_state=5)

#plt.plot(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 5,train_size=60)


linear_regression = LinearRegression().fit(x_train,y_train)
print("LinearRegression:训练集的R^2值：%f " % linear_regression.score(x_train,y_train))
print("LinearRegression:测试集的R^2值：%f " % linear_regression.score(x_test,y_test))

lasso = Lasso(selection='cyclic').fit(x_train,y_train)
print("Lasso:训练集的R^2值：%f " % lasso.score(x_train,y_train))
print("Lasso:测试集的R^2值：%f " % lasso.score(x_test,y_test))

ridge = Ridge().fit(x_train,y_train)

print("Ridge:训练集的R^2值：%f " % ridge.score(x_train,y_train))
print("Ridge:测试集的R^2值：%f " % ridge.score(x_test,y_test))


from sklearn.metrics import r2_score
#r2_score(y_train,lasso.predict(x_train))
print("r2_score:lasso测试集的R^2值：%f " % r2_score(y_train,lasso.predict(x_train)))
from sklearn.neighbors import KNeighborsClassifier

'''

'''
from sklearn.datasets import make_classification
x_class,y_class = make_classification(n_samples=100,n_features=5,n_classes=2,random_state=1)
plt.plot(x_class)
#model = KNeighborsClassifier.fit(x_train,y_train)