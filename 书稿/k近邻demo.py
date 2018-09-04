# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets  as ds
import sklearn.neighbors as nb


import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
#%% 寻找最近邻
import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[0, 0], [1, 0], [0, 1], [2, 1], [1, 2], [4, 0]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print('最近的2个近邻的索引\n',indices)
print('最近的2个近邻的距离\n',distances)

#相邻点之间的连接
nbr_graph = nbrs.kneighbors_graph(X)
print('相邻点之间的连接')
print(nbr_graph.toarray())

#%% KDTree 与 BallTree
import numpy as np
from sklearn.neighbors import KDTree,BallTree

X = np.array([[0, 0], [1, 0], [0, 1], [2, 1], [1, 2], [4, 0]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
kdt.query(X, k=3, return_distance=False) 
bt = BallTree(X, leaf_size=30, metric='euclidean')
bt.query(X, k=3, return_distance=True) 

#%% 最近邻分类 01
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from itertools import cycle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import sklearn.metrics as ms 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp

n_neighbors = 15
#%% 函数定义区域

# 通过交叉验证模型计算查准率、查全率和F1得分
def strformat(num):
    return str(round( 100 * num.mean(),2 )) + '%'

# 绘制ROC和AUC图
def draw_roc_auc(ax,clf,X,y,title):    
    # 分割训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        random_state=0)
    
    n_classes = y.shape[1]
    
    # 1对多分类器
    oneVsRestclassifier = OneVsRestClassifier(clf)
    y_score = oneVsRestclassifier.fit(X_train, y_train).predict_proba(X_test)
    
    # 计算ROC曲线和面积
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = ms.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = ms.auc(fpr[i], tpr[i])
    # 计算 micro-average ROC曲线和面积
    fpr["micro"], tpr["micro"], _ = ms.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = ms.auc(fpr["micro"], tpr["micro"])
    
    lw = 2
    
    # 计算 macro-average ROC曲线和面积，按照类别依次进行计算指标，求平均值
    # 该方法不考虑类别的不均衡
    # 首先汇总全部的fpr
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # 插值计算所有的ROC曲线上的点
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # 计算AUC 
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = ms.auc(fpr["macro"], tpr["macro"])
    
    # 绘制ROC曲线
    ax.plot(fpr["micro"], tpr["micro"],
        label='micro-average ROC曲线 (area = {0:0.2f})' 
        ''.format(roc_auc["micro"]),
        color='deeppink', linestyle=':', linewidth=4)
    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC曲线 (area = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='类{0}的ROC曲线 (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    ax.plot([0, 1], [0, 1], 'k--', lw=lw)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title,fontproperties=myfont)
    ax.legend(loc="best",prop=myfont)
    plt.show()
    return
#%% 
# 导入数据
iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target

# 网格的密度
h = .02  

# 颜色映射
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                            
fig1 = plt.figure(figsize=(10,6))
fig2 = plt.figure(figsize=(10,6))
i = 1
for weight in ['uniform', 'distance']:
    # 生成KNeighborsClassifier类的实例，并训练
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight)
    clf.fit(X, y)
    
    # 添加子图
    ax = fig1.add_subplot(1,2,i)
    
    # 依据决策边界对网格[x_min, x_max]x[y_min, y_max]间的点着色    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # 计算分类准确度
    score = clf.score(X, y)

    # 着色
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 绘制散点图
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title("k = %i, weights = '%s', score = %.2f"
              % (n_neighbors, weight,score),fontproperties=myfont)
    
    #通过交叉验证模型计算查准率、查全率和F1得分
    num_validations = 10    
    accuracy = cross_val_score(clf,X,y,scoring='accuracy',cv=num_validations)
    print('Accuracy:',strformat(accuracy))
    
    precision = cross_val_score(clf,X,y,scoring='precision_weighted',cv=num_validations)
    print('precision:',strformat(precision))
    
    recall = cross_val_score(clf,X,y,scoring='recall_weighted',cv=num_validations)
    print('recall:',strformat(recall))
    
    f1 = 2 * precision * recall/(recall + precision)
    print('F1:',strformat(f1))
    
    ## 绘制ROC和AUC图
    # 标签二值化
    y_label = label_binarize(y, classes=[0, 1, 2])
    ax2 = fig2.add_subplot(1,2,i)
    title = ' weights = %s'%(weight)
    draw_roc_auc(ax2,clf,X,y_label,title)
    
    i +=1

plt.show()

#%% 最近邻回归
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
from sklearn import neighbors
from sklearn.metrics import r2_score

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# 添加噪音
y[::5] += 1 * (0.5 - np.random.rand(8))

# 训练模型
n_neighbors = 15
fig = plt.figure()
for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)
    
    r2 = r2_score(T,y_)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='真实值')
    plt.plot(T, y_, c='g', label='预测值')
    plt.axis('tight')
    plt.legend(loc='best',prop=myfont)
    plt.title("KNeighborsRegressor (k = %i, weights = '%s', R2 = %.2f)" 
              % (n_neighbors, weights,r2))                         

plt.show()