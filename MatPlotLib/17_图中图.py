import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
x = [1,2,3,4,5,6,7]
y = [1,3,4,2,5,8,6]
# 大图
left,bottom,width,height = 0.1,0.1,0.8,0.8
ax1 = fig.add_axes([left,bottom,width,height])
ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('tithle')

# 小图
left,bottom,width,height = 0.2,0.6,0.25,0.25
ax2 = fig.add_axes([left,bottom,width,height])
ax2.plot(x,y,'r')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('tithle')

# 小图
left,bottom,width,height = 0.62,0.2,0.25,0.25
ax3 = fig.add_axes([left,bottom,width,height])
ax3.plot(y[::-1],x,'g') # y值逆序
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('tithle')


plt.show()