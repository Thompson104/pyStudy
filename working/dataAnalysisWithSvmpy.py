import  numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import  sklearn.svm as svm
import sklearn.metrics as metrics
# 加载数据

data = np.loadtxt('./iris/iris.data',delimiter=',',usecols=(0,1,2,3))
labels = np.loadtxt('./iris/iris.data',dtype='str',delimiter=',',usecols=(4))

# 构造样品数据

train_data = np.concatenate((data[0:10,], data[50:60,],data[100:110,]),axis=0)
train_label = np.concatenate((labels[0:10,], labels[50:60,],labels[100:110,]),axis=0)

# 构造测试数据,数据垂直组合concatenate
test_data = np.concatenate((data[10:50,], data[60:100,],data[110:150,]),axis=0)
test_label_expected = np.concatenate((labels[10:50,], labels[60:100,],labels[110:150,]),axis=0)
test_label_predicted = []

# 分析数据
# 创建分类器
classifier = svm.SVC(gamma='auto',kernel='rbf',C=1.0)

# 训练分类器
classifier.fit(train_data,train_label)

# 预测
test_label_predicted = classifier.predict(test_data)

# 比较结果
size = len(test_label_predicted)
outer = np.zeros((size),dtype=int)
for i in range(size):
    if test_label_expected[i] != test_label_predicted[i]:
        outer[i] = 1
result = np.vstack((test_label_expected,test_label_predicted,outer))
result = result.T

# 计算正确率
okresult = float(np.sum(outer==0)) / len(outer)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(test_label_expected, test_label_predicted)))
#result = np.concatenate((test_label_expected,test_label_predicted,outer),axis=1)

#绘制图形
plt.plot(outer,'*')