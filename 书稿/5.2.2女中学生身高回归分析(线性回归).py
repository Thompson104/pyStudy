# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:12:49 2018

@author: TIM
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%% 定义绘图辅助函数
def plt_helper(label,title):
    fig =plt.figure()
    ax = fig.add_subplot(111,label=label)
    ax.set_title(title,fontproperties=myfont)
    ax.set_xlabel('身高（cm）',fontproperties=myfont)
    ax.set_ylabel('体重（kg）',fontproperties=myfont)
    ax.axis([155,180,55,68])
    ax.grid(True)
    return ax
#%% 准备数据
X = [158, 159, 160, 161, 162, 165, 171, 172, 176]
Y = [56,  57,  58,  64,  59,  63,  62,  65,  66]
X = np.array(X)
Y = np.array(Y)

#%% 分析数据
ax = plt_helper('ax1','中学生身高与体重数据')
ax.plot(X,Y,'r*')

#%% 
from sklearn.linear_model import LinearRegression
linear_rg = LinearRegression()
linear_rg.fit( X.reshape(-1,1),Y.reshape(-1,1))

x = np.array([150])
yy = linear_rg.predict(x.reshape(-1,1))

print('预测身高为155cm的中学生的体重：%.2f kg' % yy)

print("模式的参数")
print("线性模型的截距：",linear_rg.intercept_)
print("线性模型的系数：",linear_rg.coef_)


#借助这个模型我们可以预测不同身高的中学生的体重
X2 = np.array([155, 164, 167, 180])
Y2 = linear_rg.predict(X2.reshape(-1, 1))

ax2 = plt_helper('ax2','预测不同身高的中学生的体重')
ax2.plot(X,Y,'r*')
ax2.plot(X2,Y2,'y-')

# 对模型进行评估
import sklearn.metrics as ms
yy = linear_rg.predict(X.reshape(-1,1))

print('模型的平均绝对误差 =',ms.mean_absolute_error(yy,Y))
print('模型的均方根误差 =',ms.mean_squared_error(yy,Y))
print('模型的中位数绝对误差 =',ms.median_absolute_error(yy,Y))
print('模型的解释方差 =',ms.explained_variance_score(yy,Y))
print('模型的R2值 =',ms.r2_score(yy,Y))

# 模型的残差图
ax3 = plt_helper('ax3','回归模型的残差图')
for idx,x in enumerate(X):
    ax3.plot([x,x],[yy[idx],Y[idx]],'r-')
ax3.plot(X2,Y2,'b-')  
ax3.plot(X,Y,'b.')

# 计算残差平方和
print('残差平方和: %.2f' % np.mean((yy - Y) ** 2))
#%%
'''
linear_rg_y = linear_rg.predict([155])
#plt.plot(X,linear_rg_y.T,'m-',label='regression')
print('LinearRegression均方误差=',np.mean(linear_rg_y - Y))
print("LinearRegression的score=",linear_rg.score( [X],[Y] ) )

#for (x,y,yy) in zip(X,Y,linear_rg_y):
#    plt.plot((x,x),(y,linear_rg_y),'ro')
'''


##%%拟合，自由度为freedeg
#freedeg = 1
#reg = np.polyfit(X,Y,deg=freedeg)
#ry = np.polyval(reg,X)
#plt.plot(X,ry,'r-',label='polyfit')
## 生成多项式对象
#pl = np.poly1d(reg)
#print(reg)
#print(pl) # 打印多项式
'''
for (x,y,yy) in zip(X,Y,linear_rg_y):
    plt.plot((x,x),(y,yy),'y-')
'''

'''
plt.legend(loc='best')
plt.show()
'''