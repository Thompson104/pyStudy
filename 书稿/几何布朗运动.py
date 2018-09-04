# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:27:57 2018

@author: TIM
"""

import numpy as np
import scipy.stats as scs
import matplotlib.pyplot as plt
import matplotlib as mpl

def gen_path(s0,r,sigma,T,M,I):
    dt = float(T) /M
    paths = np.zeros((M+1,I),np.float64)
    paths[0] = s0
    for t in range(1,M+1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t-1] * np.exp(( t - 0.5 * sigma ** 2) * dt +
             sigma * np.sqrt(dt) * rand)
    return paths

s0=100.
r=0.05
sigma = 0.2
T = 1.0
M = 50
I = 25000

paths = gen_path(s0,r,sigma,I,M,I)
plt.plot(paths[:,:10])
plt.grid(True)
plt.show()