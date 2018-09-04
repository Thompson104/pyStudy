# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 12:08:55 2018

@author: TIM
"""

import numpy as np
import pandas as pd

import sklearn.datasets as ds
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid",{"font.sans-serif":['simhei','Droid Sans Fallback']})
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

n_features = 10
X,y = ds.make_classification(n_samples=1000
                             ,n_features=n_features
                             ,n_clusters_per_class=1
                             , n_redundant=1
                             ,n_classes=3
                             ,random_state=1
                             )


df = pd.DataFrame(np.hstack( (X,y[:,None]) ),columns=(list(range(n_features))+['类别']))
# 指定分类变量为'类别'
#sns.set(style="ticks", color_codes=True)
_ = sns.pairplot(df[:500],
                 vars=[0,1,2,3,4],
                 hue='类别',
                 palette='husl',
                 size=1.5,
                 markers=["o", "s",'D'])
plt.show()
