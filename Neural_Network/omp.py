# -*- coding: utf-8 -*-
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DCT基作为稀疏基，重建算法为OMP算法 ，图像按列进行处理
# 参考文献: 任晓馨. 压缩感知贪婪匹配追踪类重建算法研究[D].
#北京交通大学, 2012.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 导入所需的第三方库文件
import  numpy as np
from PIL import Image #使用PIL(Python Imaging Library)，Pillow是PIL的一个派生分支

#读取图像，并变成numpy类型的 array
im = Image.open('z:\\lenna.bmp')
print('z:\\lenna.bmp', im.format, "%dx%d" % im.size, im.mode)
im = np.array(Image.open('z:\\lenna.bmp')) #图片大小256*256


#生成高斯随机测量矩阵,高斯分布即正态分布(感知矩阵Φ)
#正态分布（Normal distribution）又名高斯分布（Gaussian distribution）
sampleRate=0.5  #采样率
N = 256
M = int(sampleRate * N)
Phi=np.random.randn(M,N)  #感知矩阵：高斯随机矩阵、贝努力矩阵等，Φ（Phi） 为大小M×N矩阵

#生成稀疏基DCT矩阵，ψ为稀疏基矩阵
#稀疏基其实就是指的某种正交变换的变换矩阵列向量组成的基
mat_dct_1d=np.zeros((N,N))
v=range(N)
for k in range(0,N):
    dct_1d=np.cos(np.dot(v,k * np.pi/N))
    if k>0:
        dct_1d=dct_1d - np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d / np.linalg.norm(dct_1d) #np.linalg.norm范数

#随机测量
img_cs_1d=np.dot(Phi,im)

#OMP算法函数
def cs_omp(y,D):
    L=int(np.floor(3*(y.shape[0])/4))
    residual=y  #初始化残差
    index=np.zeros((L),dtype=int)
    for i in range(L):
        index[i]= -1
    result=np.zeros((N))
    for j in range(L):  #迭代次数
        product=np.fabs(np.dot(D.T,residual))
        pos=np.argmax(product)  #最大投影系数对应的位置
        index[j]=pos
        my=np.linalg.pinv(D[:,index>=0]) #最小二乘,看参考文献1
        a=np.dot(my,y) #最小二乘,看参考文献1
        residual=y-np.dot(D[:,index>=0],a)
    result[index>=0]=a
    return  result

#重建
sparse_rec_1d=np.zeros((N,N))   # 初始化稀疏系数矩阵
Theta_1d=np.dot(Phi,mat_dct_1d)   #测量矩阵乘上基矩阵
for i in range(256):
    # print(unicode("请输入销售额", encoding="utf-8"))
    # print('正在重建第', i, '列。')
    print("%s%d%s"%('正在重建第',i , '列。'))
    column_rec=cs_omp(img_cs_1d[:,i],Theta_1d) #利用OMP算法计算稀疏系数
    sparse_rec_1d[:,i]=column_rec;
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵

#显示重建后的图片
image2=Image.fromarray(img_rec)
# image2.save("z:\\lena_out.bmp")
image2.show()