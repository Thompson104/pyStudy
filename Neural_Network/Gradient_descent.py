# -*- coding: utf-8 -*-
# 梯度下降算法
# http://www.cnblogs.com/eczhou/p/3951861.html
# http://blog.csdn.net/woxincd/article/details/7040944
# http://blog.csdn.net/zbc1090549839/article/details/38149561
Matrix_A = [ [1,4],[2,5],[5,1],[4,2] ]
Matrix_Y = [ 19,26,19,20]
theta = [5,2]
#学习速率
learning_rate = 0.0001
loss = 50
iters = 1
Eps = 0.0001
# 随机梯度下降算法
while loss > Eps and iters < 10000:
    loss = 0
    for i in range(4):
        h = theta[0]* Matrix_A[i][0] + theta[1] * Matrix_A[i][1]
        theta[0] = theta[0] + learning_rate * (Matrix_Y[i] - h) * Matrix_A[i][0]
        theta[1] = theta[1] + learning_rate * (Matrix_Y[i] - h) * Matrix_A[i][1]
    for i in range(4):
        Error = 0
        Error = theta[0] * Matrix_A[i][0] + theta[1] * Matrix_A[i][1] - Matrix_Y[i]
        Error = Error * Error
        loss = loss + Error
    iters = iters + 1
print('theta = ',theta)
print('iters = ',iters)


