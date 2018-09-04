'''
==================
RBF SVM parameters
交叉验证，选取超参数问题
==================
'''
# ###交叉验证，选取超参数问题
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

from sklearn.model_selection import StratifiedShuffleSplit#分层洗牌分割交叉验证
from sklearn.model_selection import GridSearchCV


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

##############################################################################
# Load and prepare data set
#
# dataset for grid search

raw_data = np.loadtxt('./input/bupa.data',delimiter=',')
X = raw_data[:,:-1]
y = raw_data[:,-1]
# 绘制热力图时，用两维特征进行训练和计算（便于用网格）
X_2d = X[:, [2,5]]
y_2d = y

# 绘制散点图，观察不同维度之间的可分性
plt.figure()

for i in np.arange(0,X.shape[1]):
    for j in np.arange(i,X.shape[1]):
        if i == j :
            # plt.subplot(5, 6, (i*5+j+1))
            continue
        plt.subplot(5,6,(i*5+j+1))
        plt.scatter(X[:,i],X[:,j],c=y)
        plt.title('%d vs %d'%(i,j))


scaler = StandardScaler()#进行标准化，通过删除均值和单位方差缩放标准化功能
X = scaler.fit_transform(X)#先fit,再transform
#有信息无监督转换指只利用特征的统计信息的转换，统计信息包括均值、标准差、边界等等，比如标准化、PCA法降维等。
X_2d = scaler.fit_transform(X_2d)

##############################################################################
# Train classifiers
#

C_range = np.logspace(-2, 3, 15)# logspace(a,b,N)把10的a次方到10的b次方区间分成N份
gamma_range = np.logspace(-3, 3, 15)
param_grid = dict(gamma=gamma_range, C=C_range)
print(param_grid)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)#基于交叉验证的网格搜索。

grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))#找到最佳超参数

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range =[1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
i = 0
for C in C_2d_range:
    for gamma in gamma_2d_range:
        print('C = %f , gamma = %f'%(C,gamma))
        i=i+1
        print('i=%d'%(i))
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

##############################################################################
# visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')

    # visualize parameter's effect on decision function
    #可视化参数对决策函数的影响
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)#对网格进行画图
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')
#返回交叉验证的平均测试值，写成（len(C_range),len(gamma_range)）的形式
                                                     
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.
#绘制热力图imshow

plt.figure(figsize=(8, 6))#创建一个宽8英寸、高6英寸的图
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()