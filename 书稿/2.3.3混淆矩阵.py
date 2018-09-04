# -*- coding: utf-8 -*-
import itertools
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%% 绘制混淆矩阵的函数
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
    plt.colorbar()
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
#%% 导入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names
# 分割训练与测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#%% 建立分类器，进行训练与预测
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='混淆矩阵, 未进行归一化处理')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='混淆矩阵, 进行了归一化处理')
plt.show()
