import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc',size=10)
# ==============================================
# 1、观察数据
def runplt():
    plt.figure()
    plt.title('匹萨价格与直径数据',fontproperties=font)
    plt.xlabel('直径（英寸）',fontproperties=font)
    plt.ylabel('价格（美元）',fontproperties=font)
    # plt.axes([0,25,0,30])
    plt.grid(True)
    return plt
plt = runplt()
x = [[8],[9],  [11],[12],[16]]
y = [[11],[8.5],[15],[11],[18]]
plt.plot(x,y,'k*')
plt.show()

# ====================================================
# 2、线性回归

# 创建模型
model = LinearRegression()
# 拟合模型
model.fit(x,y)
# 回归预测
result = model.predict(12)
print('预测一张12英寸匹萨价格:%.2f' % result)
print(model)

# ===============================
# 绘制回归直线
plt = runplt()
plt.plot(x,y,'k.')
y2 = model.predict(x)
plt.plot(x,y2,'g-')

# 残差预测值
for i,xx in enumerate(x):
    # 绘制直线
    plt.plot([xx,xx],[ y[i],y2[i] ] ,'r-')
plt.show()
# 计算残差平方和：residual sum of squares
rss = (y - y2)**2

# 计算R方,
# R方是0.6620说明测试集里面过半数的价格都可以通过模型解释。
model.score(x,y)