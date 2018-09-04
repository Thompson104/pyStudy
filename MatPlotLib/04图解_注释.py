'''
图解
'''
import numpy as np
from matplotlib import pyplot as plt

x= np.linspace(-3,3,50)
y = 2*x + 1
plt.figure(num=1,figsize=(8,5))
plt.plot(x,y)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))

ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
# 添加图解
# 用散点图的形式，显示（1，0）点
x0 = 1
y0 = 2*x0 + 1
plt.scatter(x0,y0,s=50,color='b')

plt.plot([x0,x0],[y0,0],'k--',lw=2.5) #注意两个端点坐标是(x1,x2),(y1,y2)的形式给出的

# 第1种添加注释的方法
plt.annotate(r'$2x + 1 = %s$' % y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30),textcoords='offset points',
             fontsize=15,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.2'))
# plt.annotate(r'$2x + 1 = %s$' % y0,xy=(x0,y0),xycoords='data',xytext=(+30,-30), textcoords='offset point',
#              fontsize=5,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=2.'))

# 第二种添加注释的方法
plt.text(-3.7,3, r'$this\ is\ the some text. \mu \sigma_i \alpha_t$',
         fontdict={'size':16,'color':'r'})
plt.show()