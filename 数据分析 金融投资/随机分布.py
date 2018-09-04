# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 08:52:48 2017

@author: TIM
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
zt = 1.0
u = 0
xn = np.linspace(-10,10,500)
yn = 1/(np.sqrt( 2 * np.pi ) * zt) * np.e ** (-(xn -u)**2 / (2 * zt ** 2))

plt.plot(xn,yn)

# 正态分布
def test_norm_pmf():
    mu = 0
    sigma = 1
    x = np.arange(-5,5,0.1)
    y = stats.norm.pdf(x,0,1)
    print(y)
    plt.plot(x,y)
    plt.title('Normal:  $\mu $ =%.1f, $\sigma$ = %.1f '%(mu,sigma))
    return
test_norm_pmf()