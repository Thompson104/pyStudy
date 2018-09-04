import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc',size=10)
# raw_data = spi.loadmat(r'../input/XJlaman_4_1.mat')

df = pd.read_csv(r'../input/winequality-red.csv',sep=';')
# =============================================
# 1、观察数据
# print(df.head())
# print(df.describe())
# plt.scatter(df['chlorides'],df['alcohol'])
# plt.show()
# 观察相关系数
# print(df.corr())

# ==============================================
# 2、切分数据
x = df[ list(df.columns)[:-1] ]# 不包括最后一列，即‘quality’
y = df['quality']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.8)

# ===============================================
# 3、训练模型
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_predictions = regressor.predict(x_test)
print('R方',regressor.score(x_test,y_test))

# 5折交叉验证
scores = cross_val_score(regressor,x,y,cv=5)
print(scores.mean(),scores)

plt.scatter(y_test, y_predictions)
plt.xlabel('实际品质',fontproperties=font)
plt.ylabel('预测品质',fontproperties=font)
plt.title('预测品质与实际品质',fontproperties=font)
plt.show()