import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10,0.1)
y1 = 0.05* x ** 2
y2 = -1 * y1

# 没有数据的空subplot
fig,ax1 = plt.subplots()
# create a twin of Axes for generating a plot
# with a sharex x-axis but independent y axis.
ax2 = ax1.twinx()
ax1.plot(x,y1,'g--')
ax2.plot(x,y2,'b--')

plt.show()