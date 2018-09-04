import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

x = np.linspace(0,10,1000)
y = np.sin(x**2)
# fig = plt.figure(figsize=(10,5))
lines = plt.plot(x,y)
plt.ylabel('times$sin(x^2)$')
fig = plt.gcf()
axes = plt.gca()
axis = plt.gca().yaxis
axis.get_label().set_fontsize(16)
plt.ylabel('y label',fontsize=16,color='r')


for line in axis.get_ticklines():
    line.set_color('r')
    line.set_markersize(25)
    line.set_markeredgewidth(5)

for label in axis.get_ticklabels():
    label.set_color('r')
    label.set_rotation(45)
    label.set_fontsize(16)
    
plt.xticks(fontsize=16,color='g',rotation=45)
plt.show()