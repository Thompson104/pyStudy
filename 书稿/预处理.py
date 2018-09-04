
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
myfont_title = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

import numpy as np
from sklearn.preprocessing import Imputer


messing = [  [1,'NaN',4],
             [2,7,  7],
             [3,7,  5],
             [4,3,  2],
             [5,2,  2],
             [5,2,  2]]
strategys = ['mean', 'median', 'most_frequent']
for strategy in strategys:    
    model =Imputer(missing_values='NaN',strategy=strategy,axis=0,verbose=1)
    messing_imputed = model.fit_transform(messing)
    print(messing_imputed)

import sklearn.datasets as ds
import pandas as pd
raw_data = ds.load_wine()
x = raw_data.data

x = pd.DataFrame(x,columns=raw_data.feature_names)
x.describe()
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
myfont_title = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline

fig_size = (9, 6)
X, y= load_wine(return_X_y=True)
# 将数据集切分为训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=0)
# 使用没有预处理的数据训练模型.
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(X_train, y_train)
pred_test = unscaled_clf.predict(X_test)
# 使用预处理之后的数据训练模型
std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
pred_test_std = std_clf.predict(X_test)
# 对比结果.
print('\n未进行标准化处理的模型预测精度')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))
print('\n未行标准化处理的模型预测精度')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))
# 提取pca模型
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

# 对X_train data进行预处理和提取主元，用于显示
scaler = std_clf.named_steps['standardscaler']
X_train_std = pca_std.transform(scaler.transform(X_train))
# 对比标准化效果
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=fig_size)
for index, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(X_train[y_train == index, 0], X_train[y_train == index, 1],
                color=c,
                label='class %s' % index,
                alpha=0.5,
                marker=m)
for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(X_train_std[y_train == l, 0], X_train_std[y_train == l, 1],
                color=c,
                label='class %s' % l,
                alpha=0.5,
                marker=m)
    ax1.set_title('PCA降维后的训练集',fontproperties=myfont)
    ax2.set_title('PCA降维后的标准化训练集',fontproperties=myfont)
for ax in (ax1, ax2):
    ax.set_xlabel('1号主成分',fontproperties=myfont)
    ax.set_ylabel('2号主成分',fontproperties=myfont)
    ax.legend(loc='upper right')
    ax.grid()
plt.tight_layout()
plt.show()




