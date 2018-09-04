import numpy as np
import matplotlib.pyplot as plt

n =12
x = np.arange(n)
# 随机生成
y1 = (1-x/float(n)) * np.random.uniform(0.5,1.0,n)
y2 = (1-x/float(n)) * np.random.uniform(0.5,1.0,n)#

plt.bar(x,y1,facecolor='#9999ff',edgecolor='white')
plt.bar(x,-y2,facecolor='#ff9999',edgecolor='white')

for xx,y in zip(x,y1):
    # ha : horizontal alignment
    plt.text(xx + 0.0 ,y + 0.05,'%.2f' % y,ha='center',va='bottom')


for xx,y in zip(x,-y2):
    # ha : horizontal alignment
    plt.text(xx + 0.0 ,y - 0.05,'%.2f' % y,ha='center',va='bottom')


plt.show()