# %% 引用相关模块与函数
import numpy as np
import seaborn as sbn
import  matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
# %% 曲线定义函数
def f(x):
    return np.sin(x) + 0.5 * x
#%% 生成数据
np.random.seed(0)
x = np.linspace(-2 * np.pi, 2 * np.pi,100) 
xx= np.concatenate( ( x.reshape(-1,1),
                         (x**2).reshape(-1,1),
                         (x**3).reshape(-1,1),
                         np.sin(x).reshape(-1,1) )
                       ,axis=1 )
y= f(x)  + 0.05 * np.random.standard_normal(len(x))
#%%
model_one = LinearRegression()
## 选择更好的基函数，可以取得更好的效果
model_one.fit(x.reshape(-1, 1),y)
predicted_y = model_one.predict( x.reshape(-1, 1) )
print('model_one的均方误差=',np.mean(predicted_y - y))
print("model_one的score=",model_one.score(x.reshape(-1, 1),y))


model_two = LinearRegression()
# 选择更好的基函数，可以取得更好的效果
model_two.fit(xx,y)
predicted_yy = model_two.predict( xx )
print('model_two的均方误差=',np.mean(predicted_yy - y))
print("model_two的score=",model_two.score(xx,y))

#%% 绘图
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.plot(x.reshape(-1, 1),predicted_y,'r*',label='预测数据')
plt.plot(x.reshape(-1, 1),y,'y.',label='原始数据')
plt.legend(loc='best',prop=myfont)
plt.subplot(122)
plt.plot(xx[:,0],predicted_yy,'r*',label='预测数据')
plt.plot(xx[:,0],y,'y.',label='原始数据')
plt.legend(loc='best',prop=myfont)
plt.show()
