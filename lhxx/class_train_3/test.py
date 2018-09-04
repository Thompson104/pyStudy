import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

x,y,true_coefficient = make_regression(n_samples=200,
                                       n_features=30,
                                       n_informative=10,
                                       noise=100,
                                       coef=True,
                                       random_state=5)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 5,train_size=60)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression().fit(x_train,y_train)

print("训练集的R^2值：%f " % linear_regression.score(x_train,y_train))
print("测试集的R^2值：%f " % linear_regression.score(x_test,y_test))

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsClassifier

#model = KNeighborsClassifier.fit(x_train,y_train)