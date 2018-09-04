import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def fun(x):
    y = np.sin(10 * np.pi * x)/x
    return y
# 观察数据
# x = np.arange(1,2,0.01)
# y = fun(x)
#
# plt.plot(x,y)
# plt.show()

# 初始化参数
c1 = 1.49
c2 = 1.49
maxgen = 50
sizepop = 10
Vmax = 0.5
Vmin = -0.5
popmax = 2
popmin = 1
pop = np.zeros(sizepop)
v = np.zeros(sizepop)
fitness = np.zeros(sizepop)

# IV 产生初始粒子和速度
for i in np.arange(0,sizepop):
    # popmin~popmax 之间
    pop[i] = (popmax -popmin) * np.random.sample() + popmin
    # -0.5 ~ 0.5
    # v[i] = 0.5 * ( 2 * np.random.sample() - 1)
    v1[i] = (Vmax - Vmin) * np.random.sample() + Vmin
    # 计算适应度
    fitness[i] = fun(pop[i])

# V 计算个体极值和群体极值
bestfitness = np.max(fitness)
# 最大元素对应的索引
# x == np.max(x) 获得一个掩模矩阵，
# 然后使用where方法即可返回最大值对应的行和列
bestindex = np.where(fitness == np.max(fitness)) # bestindex[0][0]
# bestindex = fitness.tolist().index(bestfitness)

zbest = pop[bestindex] #
gbest = pop
fitnessgbest = fitness
fitnesszbest = bestfitness

yy = np.zeros(maxgen)
# VI 迭代寻优
for i in np.arange(0,maxgen):
    for j in np.arange(0,sizepop):
        # 速度更新
        v[j] = v[j] + c1 * np.random.random() * (gbest[j] -pop[j] ) \
               + c2 * np.random.random() * (zbest - pop[j])
        if v[j] > Vmax:
            v[j] = Vmax
        elif v[j] < Vmin:
            v[j] = Vmin
        # 种群更新
        pop[j] = pop[j] + v[j]
        if pop[j] > popmax:
            pop[j] = popmax
        elif pop[j] < popmin:
            pop[j] = popmin

        # 适应度更新
        fitness[j] = fun(pop[j])

    for j in np.arange(0,sizepop):
        # 个体最优更新
        if fitness[j] > fitnessgbest[j]:
            gbest[j] = pop[j]
            fitnessgbest[j] = fitness[j]
        # 群体最优更新
        if fitness[j] > fitnesszbest:
            zbest = pop[j]
            fitnesszbest = fitness[j]

    yy[i] = fitnesszbest
    print(fitnesszbest)

# 输出结果并绘图
print(fitnesszbest,zbest)
print(yy)
plt.plot(yy)
plt.show()