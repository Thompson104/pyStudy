# #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式s
import seaborn as sns
import numpy as np
import  matplotlib as mpl
import  matplotlib.pyplot as plt
'''
seaborn将matplotlib的参数划分为两个组。第一组控制图表的样式，第二组控制图的度量尺度元素，
这样就可以轻易在纳入到不同的上下文中
操控这些参数由两个函数提供接口。
控制样式，用axes_style()和set_style()这两个函数
度量图则用plotting_context()和set_context()这两个函数
'''
# x = np.arange(-10* np.pi ,10 * np.pi,np.pi/100)
# y = np.sin(x)
flip =1


def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1,7):
        y = np.sin(x + i * 0.5) * (7 - i) * flip
        # plt.plot(x,y)
        # plt.scatter(x,y)
        plt.hist(x)

'''
seaborn目前有五种预设的样式：darkgrid（灰色网格）、whitegrid（白色网格）、dark（灰色）、white（白色）和ticks。
它们根据不同人的爱好被应用于不同的使用场景。默认的样式为darkgrid。如上面所述，网格线对于传播信息很有用，
而白灰背景样式有助于更好的展示数据。whitegrid样式类似，但它更适用于绘制带有复杂数据元素的图表，比如箱型图：
'''
sns.set_style("ticks")
# sns.set_style("dark")
sinplot()
plt.show()