# -*- coding: utf-8 -*-
import itertools
import numpy as np
import pandas as pd
from itertools import cycle
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import sklearn.metrics as ms 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from scipy import interp

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%% 加载数据
raw_data = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[raw_data['data'],raw_data['target']],
                    columns= np.append( raw_data['feature_names'],['y']))

# 观察数据格式
print( iris.head(5) )

# 观察数据是否缺失
print( iris.isnull().sum() )
# 观察样本类别是否均衡
print( iris.groupby('y').count() )

# 构造数据集
X = iris[raw_data.feature_names]
y = iris['y']
# 分割为训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=0)
class_names = raw_data.target_names
#%% 创建KNN分类模型
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#%% 通过交叉验证模型计算查准率、查全率和F1得分
def strformat(num):
    return str(round( 100 * num.mean(),2 )) + '%'

num_validations = 10    
accuracy = cross_val_score(classifier,X,y,scoring='accuracy',cv=num_validations)
print('Accuracy:',strformat(accuracy))

precision = cross_val_score(classifier,X,y,scoring='precision_weighted',cv=num_validations)
print('precision:',strformat(precision))

recall = cross_val_score(classifier,X,y,scoring='recall_weighted',cv=num_validations)
print('recall:',strformat(recall))

f1 = 2 * precision * recall/(recall + precision)
print('F1:',strformat(f1))
#%% 计算混淆矩阵
# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
    """
    功能：绘制与输出混淆矩阵.
    normalize=True，归一化.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("混淆矩阵, 进行了归一化处理")
    else:
        print('混淆矩阵, 未进行归一化处理')
    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontproperties = myfont)
#    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实标签',fontproperties=myfont)
    plt.xlabel('预测标签',fontproperties=myfont)
    pass

cnf_matrix = ms.confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
# 绘制未进行归一化处理的混淆矩阵
fig = plt.figure()
ax = fig.add_subplot(121)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='未归一化处理')
# 绘制进行了归一化处理的混淆矩阵
ax = fig.add_subplot(122)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='归一化处理')
plt.suptitle('混淆矩阵',fontproperties=myfont)
plt.show()

#%% ROC曲线和AUC指标
X,y = datasets.load_iris(return_X_y=True)
# 标签二值化
y = label_binarize(y, classes=[0, 1, 2])
# 分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.5,
                                                    random_state=0)

n_classes = y.shape[1]

# 1对多分类器
oneVsRestclassifier = OneVsRestClassifier(classifier)
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
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
    label='micro-average ROC曲线 (area = {0:0.2f})' 
    ''.format(roc_auc["micro"]),
    color='deeppink', linestyle=':', linewidth=4)
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC曲线 (area = {0:0.2f})'
         ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='类{0}的ROC曲线 (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC曲线',fontproperties=myfont)
plt.legend(loc="best",prop=myfont)
plt.show()