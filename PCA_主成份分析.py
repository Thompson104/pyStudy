print(__doc__)
'''
PCA,主成份分析
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition as dp
import sklearn.datasets as ds

raw_data = ds.load_iris()
x = raw_data.data
y = raw_data.target

pca = dp.PCA(n_components=3)
pca.fit(x)
x = pca.transform(x)

np.random.seed(1)

fig = plt.figure(1,figsize=(4,3))
plt.clf() #clears the entire current figure
ax = Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
plt.cla() # clears an axis,

for name,label in [('Setosa',0),('Versicolour',1),('Virginica',2)]:
    ax.text3D(x[y == label,0].mean(),
              x[y == label,1].mean() + 1.5,
              x[y == label,2].mean(),
              name,
              horizontalalignment='center',
              bbox = dict(alpha = 0.5,edgecolor='w',facecolor='w'))
y = np.choose(y, [0,1, 2]).astype(np.float)
# y = np.choose(y, [1,2, 0]).astype(np.float)
ax.scatter(x[:, 0], x[:, 1], x[:, 2],s=40, c=y, cmap=plt.cm.spectral)

# 取消坐标轴刻度标签
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
