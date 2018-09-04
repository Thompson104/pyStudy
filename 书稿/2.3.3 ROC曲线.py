# -*- coding: utf-8 -*-

import numpy as np
import sklearn.datasets as ds

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
# 加载iris数据集
X,y = ds.load_iris(return_X_y=True)
# 将数据集改为二分类数据集
y[y==2] = 0
# 切分数据集
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# 建立分类模型，预测测试集
model = svm.SVC(kernel='linear',probability=True,random_state=0)
model.fit(x_train,y_train)
y_score = model.predict_proba(x_test)

#计算FPR,TPR值
fpr,tpr,_ = roc_curve(y_test,y_score[:,1])
roc_auc = auc(fpr,tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr,tpr,color='darkorange',
         label='ROC 曲线 (AUC = %0.2f)' % roc_auc)
# 绘制45度参考线
plt.plot([0.0,1.0],[0.0,1.0],'g--')
plt.xlim([-0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('假正例率-FPR',fontproperties=myfont)
plt.ylabel('真正例率-TPR',fontproperties=myfont)
plt.title('ROC曲线与AUC示意图',fontproperties=myfont)
plt.legend(loc='best',prop=myfont)
plt.show()