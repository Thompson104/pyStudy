# -*- coding: utf-8 -*-
import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%% 模拟数据生成
def data_helper():
    '''
    生成用于多元线性回归分析的数据
    '''
    (X,y,coef)=ds.make_regression(n_samples=100,n_features=5,n_informative=3,
                       n_targets=1,bias=2.7,noise=2.1,coef=True)
    return (X,y,coef)

(X,y,coef) = data_helper()
