# -*- coding: utf-8 -*-
"""
Logistic_regression 
垃圾邮件分类
"""
import numpy as np
import scipy as sp
import scipy.io as spi
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.cross_validation import train_test_split
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
#=====================
# 用pandas加载数据.csv文件
df = pd.read_csv('../input/SMSSpamCollection',delimiter='\t',header=None)
# 观察数据
#print(df.head())
print('spanm:',df[df[0]=='spam'][0].count() )
print('ham:',df[df[0]=='ham'][0].count() )

#=====================================
# 用train_test_split分成训练集（75%）和测试集（25%）
x = df[1]
y = df[0]
x_train_raw,x_test_raw,y_train,y_test = train_test_split(x,y,train_size=0.75)

#===================================
# 建一个TfidfVectorizer实例来计算TF-IDF权重
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train_raw)
x_test = vectorizer.transform(x_test_raw)
# ========================================
# 建立分类器
classifier = LogisticRegression()
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)

score = classifier.score(x_test,y_test)
print('准确率：',score)
# 交叉验证
scores = cross_val_score(classifier,x_test,y_test,cv=5)
print('准确率：',np.mean(scores), scores)
results = np.array([y_test.reshape(1393).tolist(),predictions.reshape(1393).tolist()]).T

#==============================================
# 可视化分类器的效果
#http://blog.csdn.net/zdy0_2004/article/details/44948511
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y_test = labelencoder.fit_transform(y_test)
predictions_Proba = classifier.predict_proba(x_test)
false_positive_rate,recall,thresholds = roc_curve(y_test,predictions_Proba[:,1])
roc_auc = auc(false_positive_rate,recall)
plt.title('reciver operating characterstic')
plt.plot(false_positive_rate,recall,'b')
plt.grid()
plt.show()