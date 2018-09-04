# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:51:52 2018

@author: Tim
"""

import numpy as np
import matplotlib.pyplot as plt
class Perceptron():
    '''
    eta:学习率
    n_iter：权重向量的训练次数
    w_:神经元权重
    '''
    def __init__(self,eta=0.01,n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = []
        return
    def fit(self,X,y):
        '''
        X:shape[n_samples,n_features]
        y:shape[n_samples]
        '''
        return
    pass
