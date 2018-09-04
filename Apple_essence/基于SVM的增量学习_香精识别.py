# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


if __name__ == '__main__' :
    # ======================================================
    # 一、获取数据
    x = np.loadtxt('x.txt')
    y = np.loadtxt('y.txt')

    # ======================================================
    # 二、数据预处理
    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # PCA降维
    pca = PCA(n_components=10)
    pca.fit(x)
    x = pca.transform(x)

    # 切分训练集与测试集
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.2, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 训练svm模型
    ini_clf = SVC()
    model2 = ini_clf.fit(train_x, train_y)
    result_y = ini_clf.predict(test_x)
    result = result_y - test_y
    print('=====================================')
    print('初始化分类器：')
    print('壳向量集作为新的训练样本集,正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('类别：%d 的支持向量数量为%d' % (0, ini_clf.n_support_[0]))
    print('类别：%d 的支持向量数量为%d' % (1, ini_clf.n_support_[1]))

    # 3、开始增量学习
    batchs = 20  # 学习批次
    nums = np.floor(test_x.shape[0] / batchs).astype('int')  # 进行切片运算时，必须是整数
    remainder = test_x.shape[0] % batchs
    results = np.zeros((batchs, 4))
    print('=====================================')
    print('学习批次=\t', batchs)
    print('每批样本数量=\t', nums)
    print('余数=\t', remainder)
    print('=====================================')
    print('开始增量学习：')
    batchs_train_x = train_x.copy()
    batchs_train_y = train_y.copy()
    for i in np.arange(0, batchs):
        print('第%d次增量学习' % (i + 1))
        train_size = 0.8
        # 用nums *  train_size 个新增样本进行训练，剩余进行测试
        batchs_train_x = np.vstack((train_x[:,:],test_x[i * nums : ((i + 1) * nums * train_size).astype('int'), :]))
        batchs_train_y = np.concatenate((train_y[:],test_y[i * nums : ((i + 1) * nums * train_size).astype('int')]),axis=0)
        clf = SVC()
        model = clf.fit( batchs_train_x,
                         batchs_train_y
                        )
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums, :])
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        print('类别：%d 的支持向量数量为%d' % (0, clf.n_support_[0]))
        print('类别：%d 的支持向量数量为%d' % (1, clf.n_support_[1]))

        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i, :] = [i,
                         clf.n_support_[0],
                         clf.n_support_[1],
                         (np.sum(result == 0) / result.shape[0])]

    np.savetxt('results_svm.txt', results)
    # 绘图
    plt.plot(results[:, 0], results[:, -1], 'r-*')
    plt.ylim((0, 1))
    plt.show()
