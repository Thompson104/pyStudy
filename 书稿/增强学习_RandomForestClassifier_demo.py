import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=15)
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
#%% 加载乳腺癌数据集
raw_data = ds.load_breast_cancer()
X = raw_data.data
y = raw_data.target
feature_names = raw_data.feature_names
target_names = raw_data.target_names
# 切分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(X,y,
                                                 shuffle=True,
                                                stratify=y,
                                                 random_state=0)
#%% 使用默认参数训练随机森林
#clf = RandomForestClassifier()
#clf.fit(x_train,y_train)
## 输出各特征的重要性
#print('feature_importances_ = ',clf.feature_importances_)
#ind = np.argsort(clf.feature_importances_)[::-1]
#total = 0.0
#print('各特征的重要性')
#for i , feature_importance in enumerate(clf.feature_importances_[ind]):
#    if i > 9:
#        break
#    print("%-25s : %.4f"%(feature_names[i],feature_importance))
#    total += feature_importance
#    pass
#print("%-23s : %.4f"%('合计',total))
#print('')
#print("%-20s : %.4f"%('测试集得分',cross_val_score(clf,X,y).mean()))
#
## 使用10个特征进行分类
#clf.fit(x_train[:,ind[0:10]],y_train)
#print("%-20s : %.4f"%('基于10个特征进行测试集得分',cross_val_score(clf,X[:,ind[0:10]],y).mean()))


#%% 对比分析不同的n_estimators取值对分类性能的影响
#n_estimators_List = np.arange(1,150)
#f1_scores = []
#for n_estimators in n_estimators_List:
#    clf = RandomForestClassifier(n_estimators=n_estimators,
#                                 random_state=0)
#    clf.fit(x_train, y_train)
#    f1_scores.append(f1_score(y_test,clf.predict(x_test)))
## 绘制同的n_estimators取值对分类性能的影响
#fig = plt.figure(figsize=(6,4))
#ax = fig.add_subplot(111)
#ax.plot(f1_scores)
#ax.grid(True)
#ax.set_xlabel('决策树的数量',fontproperties=myfont)
#ax.set_ylabel('F1得分',fontproperties=myfont)
#ax.set_title('决策树的数量对随机森林分类性能的影响',fontproperties=myfont)
#fig.show()


#%% 对比分析不同的决策树大小对随机森林分类器性能的影响

min_samples_leaf_List = np.arange(1,20)
f1_scores = []
for min_samples_leaf in min_samples_leaf_List:
    clf = RandomForestClassifier(min_samples_leaf = min_samples_leaf,
                                 random_state=0)
    clf.fit(x_train, y_train)
    f1_scores.append(f1_score(y_test,clf.predict(x_test)))
    print('')
    
# 绘制同的n_estimators取值对分类性能的影响
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(f1_scores)
ax.grid(True)
ax.set_xlabel('决策树的数量',fontproperties=myfont)
ax.set_ylabel('F1得分',fontproperties=myfont)
ax.set_title('决策树的数量对随机森林分类性能的影响',fontproperties=myfont)
fig.show()

#%% 对比分析不同的决策树大小对随机森林分类器性能的影响
'''
max_leaf_nodes_List = np.arange(2,20)
f1_scores = []
for max_leaf_nodes in max_leaf_nodes_List:
    clf = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,
                                 min_samples_leaf = 5,
                                 random_state=0)
    clf.fit(x_train, y_train)
    f1_scores.append(f1_score(y_test,clf.predict(x_test)))
# 绘制同的n_estimators取值对分类性能的影响
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
ax.plot(f1_scores)
ax.grid(True)
fig.show()
'''