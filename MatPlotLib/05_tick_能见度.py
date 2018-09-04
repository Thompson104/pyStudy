import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,3,50)
y = 0.1*x

plt.figure()
plt.plot(x,y,linewidth=10)
plt.ylim(-2,2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#调整坐标轴
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
# 讲x轴的数字label的属性
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(15)
    label.set_bbox(dict(facecolor='y',edgecolor='r',alpha=0.7))
plt.show()