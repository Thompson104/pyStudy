import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)
myfont_title = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)

from sklearn.svm import OneClassSVM
import sklearn.datasets as ds
#%%
# 加载iris数据
X,y = ds.load_iris(return_X_y=True)
# 保留样本的两个特征
X = X[:,[1,2]]

# 类别0和类别1作为正常点
X_temp = X[y!=2]
y_temp = y[y!=2]

X_train = X_temp[10:-10,:]
y_train = y_temp[10:-10,]

X_test = np.vstack(( X_temp[-10:,:],X_temp[0:10,:]))
y_test = np.vstack(( y_temp[-10:,],y_temp[0:10,]))

# 类别2为异常点
X_outliers = X[y==2]
y_2 = y[y==2]

# 绘制正常点的训练集
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
# 构建网格
xx, yy = np.meshgrid(np.linspace(X.min() - 0.1 , X.max() + 0.1, 500),
                     np.linspace(X.min() - 0.1 , X.max() + 0.1, 500))

# 构建OneClassSVM并训练
clf = OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)
clf.fit(X_train)
#预测训练集、测试集和异常数据集
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
# 预测的异常数据的数量
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# 绘制分界线
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制等高线
plt.title("异常检测",fontproperties=myfont_title)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,edgecolors='k')
plt.axis('tight')

plt.legend([a.collections[0], b1, b2, c],
    ["边界线", "训练数据", "新的正常数据", "新的异常数据"],
    loc="upper left",
    prop=myfont)
plt.xlabel("误判训练样本: %d/%d ; 误判的新正常数据: %d/%d ; "
            "误判的新异常数据: %d/%d"
            % (n_error_train, X_train.shape[0],
               n_error_test,X_test.shape[0], 
               n_error_outliers,X_outliers.shape[0]),fontproperties=myfont)
plt.show()