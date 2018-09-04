import numpy as np
from scipy import interpolate
from scipy.special import jn
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=u'C:\Windows\Fonts\simsun.ttc',size=10)

x = np.linspace(0, 10, 100)
#%% 贝塞尔函数
fig, ax = plt.subplots()
for n in range(4):
    ax.plot(x, jn(n, x), label=r"$J_%d(x)$" % n)
ax.legend()
#%% 插值
# 创建信号
plt.figure()
x = np.linspace(-18,18,36)
noise = 0.1*np.random.random(len(x))
signal = np.sinc(x) + noise

# 生成一次插值函数
interpreted = interpolate.interp1d(x,signal) #<---------
x2 = np.linspace(-18,18,180)
y = interpreted(x2)
 
# 生成三次插值函数
cubic = interpolate.interp1d(x,signal,kind='cubic') #<---------
y2 = cubic(x2)
 
plt.plot(x,signal,marker='o',label='data')
plt.plot(x2,y,linestyle='-',label='linear')
plt.plot(x2,y2,'-',lw=2,label='cubic')
plt.legend()
plt.show()

#%% 傅立叶变换
plt.figure()
#采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点，一样意思的）
x=np.linspace(0,1,1400)      

#设置需要采样的信号，频率分量有180，390和600
y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)

yy=fft(y)                     #快速傅里叶变换
yreal = yy.real               # 获取实数部分
yimag = yy.imag               # 获取虚数部分

yf=abs(fft(y))                # 取绝对值
yf1=abs(fft(y))/len(x)           #归一化处理
yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

xf = np.arange(len(y))        # 频率
xf1 = xf
xf2 = xf[range(int(len(x)/2))]  #取一半区间

plt.subplot(221)
plt.plot(x[0:50],y[0:50])   
plt.title('原始波形',fontproperties=myfont)

plt.subplot(222)
plt.plot(xf,yf,'r')
plt.title('混合波的快速傅立叶变换(双边频率范围)',
          fontproperties=myfont) 

plt.subplot(223)
plt.plot(xf1,yf1,'g')
plt.title('混合波的快速傅立叶变换(标准化处理后)',
          fontproperties=myfont)

plt.subplot(224)
plt.plot(xf2,yf2,'b')
plt.title('混合波的快速傅立叶变换',
         fontproperties=myfont)

plt.show()