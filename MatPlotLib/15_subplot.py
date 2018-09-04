import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(2,1,1)
plt.plot([0,1],[0,1],color='r')

plt.subplot(2,3,4) # 第4个小图
plt.plot([0,1],[0,1],color='y')

plt.subplot(235)
plt.plot([0,1],[0,1],color='b')

plt.subplot(2,3,6)
plt.plot([0,1],[0,1],color='c')
plt.show()