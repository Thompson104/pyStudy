from matplotlib import  pyplot as plt
import numpy as np
'''
控制坐标值,修改标签值
'''
x = np.linspace(-3,3,50)
y1 = 2*x + 1
y2 = x**2
#创建第一幅图
plt.figure(num=2,figsize=(5,4))
line1, = plt.plot(x,y1,color='red',linestyle='--',label='down') # line1后必须跟一个逗号，否则legend函数中的handles无效
line2, = plt.plot(x,y2,label='up') #设置图例名称
line3, = plt.plot(x,y1+y2,label='down')
plt.legend(handles=[line1,line2,line3],loc='best')        #显示图例upper right,best

# 设置坐标值的范围
plt.xlim((-1,2)) #默认间隔有10个
plt.ylim((-2,3))
# 设置x，y轴的坐标标签
plt.xlabel('I am x')
plt.ylabel('I am y')



# 修改坐标轴中的小标签，ticks
new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks) #从-1到2，5个间隔
# 将y轴数字，替换为字符
# plt.yticks([-2,-1.8,1,1.22,3,],
#            ['$相当差$',r'$\alpha\ \mu$','normal','good','very good']) # 用$包含的是数学字体，r表示正则表达的形式

# 修改坐标值的位置,gca = 'get current axis'
ax = plt.gca()
# 坐标轴，spines为figure四周的边框
ax.spines['right'].set_color('none')#右边框消失
ax.spines['top'].set_color('none')  #顶边框消失
ax.xaxis.set_ticks_position('bottom')   #设置x轴的ticks为left边框
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',1))   #设置x轴（底边框）位于y轴的1
ax.spines['left'].set_position(('data',0))      #设置y轴（底边框）位于x轴的0
#显示图片
plt.show()