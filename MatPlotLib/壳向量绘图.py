import numpy as np
#import scipy.io as spi
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt

x1 = np.array([1,3,5,9])
y1 = x1 * 3 + 2

x2 = np.array([1,1.4,3.6,9])
y2 = x2 * 3 + 10

x3 = np.array([1,3,5,9])
y3 = x3 * 3 + 6

x11= np.array([[3.5,12.5],[3.6,8.1],[3.68,9.23],[4.8,5.41],[5.41,16.8],[5.5,18.5],[5.75,13.78],[5.8,7.7],[4.8,10.63],[4.22,12.26]])
# x11= np.array([[3.5,12.5],[3.6,8.1],[4.8,5.41],[5.41,16.8],[5.5,18.5],[5.75,13.78],[5.8,7.7]])
x22= np.array([[1.62,29.61],[2,16],[2.28,33.78],[2.5,21.7],[3.25,34.48],[4.5,28.12],[5,25],[2.42,25.39],[3.33,24.82],[1.36,22.81]])
# x22= np.array([[2,16],          [1.62,29.61],[2.28,33.78],[3.25,34.48],[4.5,28.12],[5,25]])

hull_x11 = ConvexHull(x11)
hull_x22 = ConvexHull(x22)

fig = plt.figure()
plt.plot(x1,y1,c='r',label='f(x)=1')
plt.plot(x2,y2,c='y')
plt.plot(x3,y3,c='b')
plt.text(6.3,33,'f(x)=1')
plt.text(7.3,29,'f(x)=0')
plt.text(8.3,25,'f(x)=-1')

# index1 = np.sort(hull_x11.vertices)
# index2 = np.sort(hull_x22.vertices)
index1 = hull_x11.vertices
index2 = hull_x22.vertices

plt.scatter(x11[:,0],x11[:,1],marker='*',c='r')
plt.scatter(x22[:,0],x22[:,1],marker='s',c='y')

plt.scatter(x11[index1,0],x11[index1,1],marker='*',c='r')
plt.scatter(x22[index2,0],x22[index2,1],marker='s',c='y')

index1 = np.concatenate((index1,index1[[0,-1]]))
plt.plot(x11[index1,0],x11[index1,1],marker='*',c='r')
index2 = np.concatenate((index2,index2[[0,-1]]))
plt.plot(x22[index2,0],x22[index2,1],marker='s',c='y')

# from scipy.interpolate import spline  # 插值计算光滑曲线
# from scipy.interpolate import BSpline

# new_x22_0 = np.linspace(x22[:,0].min(),x22[:,0].max(),300)
# new_x22_1 = spline(x22[:,0],x22[:,1],new_x22_0)
# new_x22_2 = BSpline(new_x22_0,x22[:,1],)
# plt.plot(new_x22_0,new_x22_1,marker='s',c='y')
# plt.plot(x22[[9,0],0],x22[[9,0],1],marker='s',c='y')
plt.show()