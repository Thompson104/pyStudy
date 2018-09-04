# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 12:11:35 2018

@author: Tim
"""

from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
import matplotlib.pyplot as plt
import pickle as pkl
import gzip

def load_data():
    with gzip.open(r'z:\mnist.pkl.gz') as fp:
        training_data, valid_data, test_data = pkl.load(fp,encoding='iso-8859-1')
    return training_data, valid_data, test_data

def init_coefs_(X,y):
    model = BernoulliRBM(random_state=0,
                      verbose=True,
                      learning_rate=0.1,
                      n_iter=20)
    model.fit(X,y)
    return model.intercept_visible_
    

training_data, valid_data, test_data = load_data()

X_train, X_test = training_data[0][:], test_data[0][:]
y_train, y_test = training_data[1][:], test_data[1][:]
# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
# solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,40), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=0.1)
coef = init_coefs_(X_train, y_train)
print('coef=',coef.shape)
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5*vmin,vmax=.5*vmax)
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()