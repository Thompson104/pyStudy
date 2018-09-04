import numpy as np
import matplotlib.pyplot as plt


# 方法1: subplot2gird
plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),rowspan=1,colspan=3)
ax2 = plt.subplot2grid((3,3),(1,0),rowspan=1,colspan=2)
ax3 = plt.subplot2grid((3,3),(1,2),rowspan=2,colspan=1)
ax4 = plt.subplot2grid((3,3),(2,0),rowspan=1,colspan=1)
ax5 = plt.subplot2grid((3,3),(2,1),rowspan=1,colspan=1)

# 方法2：
import matplotlib.gridspec as gridspec
plt.figure()
gs = gridspec.GridSpec(3,3)
ax1 = plt.subplot(gs[0,:])
ax2 = plt.subplot(gs[1,:2])
ax3 = plt.subplot(gs[1:,2])
ax4 = plt.subplot(gs[-1,0])
ax5 = plt.subplot(gs[-1,-2])

# 方法3：简单
fig,((ax11,ax12),(ax21,ax22)) = plt.subplots(2,2,sharex=True,sharey=True)
ax11.plot([0,1],[0,1],color='r')
# ax1.plot([0,1],[0,1],color='r')
# ax1.set_title('ax1')
# ax1.set_xlim(0,1)
plt.show()