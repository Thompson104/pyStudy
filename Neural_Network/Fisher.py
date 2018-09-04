import numpy as np
import scipy as sp
# csv文件整体读取为一个矩阵
data = np.loadtxt(open('z:\\FisherData.csv','r'),delimiter=',')
x1 = data[0:6,]
x2 = data[6:,]
# 每列为一个样本
x1 = np.transpose(x1)
x2 = np.transpose(x2)

# 计算每一维度的平均值，这里是行
m1 = sp.mean(x1,axis=1)
m2 = sp.mean(x2,axis=1)

# 计算两类样品类内离散矩阵
# np.cov :each column a single  observation of all those variables.
# 每一列代表一个样本或观测，与matlab的cov函数是每一行代表一个观测
# py -3 -m pydoc -b
s1 = np.cov((x1 - m1))
s2 = sp.cov((x2 - m2))

# 总类内离散矩阵
Sw = s1 + s2

# 类间离散度矩阵
Sb =  np.dot((m1 - m2) ,np.transpose(m1 - m2))
print(Sb)