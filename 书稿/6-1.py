from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
myfont_title = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

n_train = 20 # samples for training
n_test = 200 # samples for testing
n_averages = 50 # how often to repeat classification
n_features_max = 75 # maximum number of features
step = 4 # step size for the calculation
def generate_data(n_samples, n_features):
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])
#    print(X)
    # add non-discriminative features
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
        pass
#    print(X.shape)
    return X, y

acc_clf1, acc_clf2 = [], []
n_features_range = range(1, n_features_max + 1, step)
for n_features in n_features_range:
    score_clf1, score_clf2 = 0, 0
    for _ in range(n_averages):
        X_train, y_train = generate_data(n_train, n_features)
#        print(X_train.shape)
        clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X_train, y_train)
        clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)
    
        X_test, y_test = generate_data(n_test, n_features)
        score_clf1 += clf1.score(X_test, y_test)
        score_clf2 += clf2.score(X_test, y_test)
        pass
       
    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)
    pass
   
features_samples_ratio = np.array(n_features_range) / n_train
plt.plot(features_samples_ratio, acc_clf1,linestyle='-', linewidth=2,label="收缩LDA", color='black')
plt.plot(features_samples_ratio, acc_clf2,linestyle='--', linewidth=2,label="LDA", color='black')

plt.xlabel('特征数量 / 样本数量',fontproperties=myfont)
plt.ylabel('分类准确度',fontproperties=myfont)
plt.legend(loc='best', prop=myfont)
#plt.suptitle('lDA与收缩LDA对比')
plt.show()
