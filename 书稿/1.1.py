#%% 引用模块与函数
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=16)
#%% 生成数据
np.random.seed(0)
x  =  np.linspace( -2  *  np.pi,  2  * np.pi, 100) 
y  =  3.5 * x  +   2  *  np.random.standard_normal(len(x))
# 分割数据
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=5)

#%% 
#创建回归模型
model = LinearRegression()
# 训练回归模型
model.fit(train_x.reshape(-1, 1),train_y)
# 使用测试数据进行测试
test_yy = model.predict(test_x.reshape(-1, 1))
yy = model.predict(x.reshape(-1, 1))
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
std_res_yy = scaler.fit_transform((yy-y).reshape(-1, 1))

#%% 绘图观察结果
plt.figure()
plt.tight_layout()
plt.subplot(311)
plt.plot(train_x,train_y,'r*',label='训练数据')
plt.plot(test_x,test_yy,'y-',label='拟合数据')
plt.plot(test_x,test_y,'m.',label='测试数据')
plt.legend(loc='best',prop=myfont)
plt.subplot(312)
# 残差图
tempy = test_yy - test_y
#plt.scatter(x,yy-y,c='r')
plt.plot((-7,9),(2,2),'r-')
plt.scatter(x,std_res_yy,c='b')
plt.plot((-7,9),(-2,-2),'r-')

plt.subplot(313)
for (x,y,yy) in zip(test_x,test_y,test_yy):
    plt.plot((x,x),(y,yy),'b-')
plt.plot(test_x,test_yy,'r-')
plt.scatter(test_x,test_y,marker='o')
plt.title('残差图',fontproperties=myfont)
plt.show()
