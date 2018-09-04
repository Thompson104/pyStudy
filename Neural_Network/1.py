# -*- coding: utf-8 -*-
import numpy as np

def nonlin(x,deriv=False):
    a = x*(1-x)
    b = 1/(1 + np.exp(-x))
    if(deriv == True):
        return a
    return b

X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1]])

y = np.array([[0,0,1,1]]).T
np.random.seed(1)
syn0 = 2*np.random.random((3,1)) - 1

L0 = X
for iter in np.arange(10000):
    L1 = nonlin(np.dot(L0,syn0))
    L1_error = y - L1

    L2 = nonlin(L1,True)
    L1_delta = L1_error * L2
    #L1_delta = np.dot(L1_error,L2)
    syn1 = np.dot(L0.T,L1_delta)
    syn0 = syn1 + syn0

print(L1)

