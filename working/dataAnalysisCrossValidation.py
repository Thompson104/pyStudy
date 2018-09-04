import  numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import  sklearn.neighbors as nb
import sklearn.metrics as metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score
# 中文显示
zhfont = mp.font_manager.FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=14) #宋体常规

# 加载数据
iris = datasets.load_iris()

# 构造样品数据

train_data,test_data,train_label,test_label_expected = \
                        train_test_split(iris.data,iris.target,test_size=0.51,random_state=0)



# 分析数据
# 创建分类器
#classifier = nb.KNeighborsClassifier(n_neighbors=3,weights='uniform',algorithm='auto')
classifier = nb.RadiusNeighborsClassifier(n_neighbors=2,weights='uniform',algorithm='auto')

# 训练分类器
classifier.fit(train_data,train_label)

# 预测
test_label_predicted = classifier.predict(test_data)
# 交叉验证
scores = cross_val_score(classifier,iris.data,iris.target,cv=10)

# 比较结果
size = len(test_label_predicted)
outer = np.zeros((size),dtype=int)
for i in range(size):
    if test_label_expected[i] != test_label_predicted[i]:
        outer[i] = 1
result = np.vstack((test_label_expected,test_label_predicted,outer))
result = result.T

# 计算正确率
#classifier.score(test_data,test_label_expected)
okresult = float(np.sum(outer==0)) / len(outer)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(test_label_expected, test_label_predicted)))
#result = np.concatenate((test_label_expected,test_label_predicted,outer),axis=1)

#绘制图形
plt.plot(outer,'*')
plt.xlabel(u"样板编号",fontproperties=zhfont)
plt.ylabel(u'y=0 正确分类 或 y= 1 分类错误',fontproperties=zhfont)
plt.title(u'分类正确',fontproperties=zhfont)
plt.show()