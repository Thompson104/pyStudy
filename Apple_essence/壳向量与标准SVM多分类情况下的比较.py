# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
# -*- coding: utf-8 -*-
"""
基于壳向量的线性支持向量机快速增量学习算法
"""
import sys
import os
import time
import numpy as np
# import scipy.io as spi
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
# from sklearn import cross_validation as cv
This module was deprecated in version 0.18 in favor of the model_selection module 
into which all the refactored classes and functions are moved. Also note that 
the interface of the new CV iterators are different from that of this module. 
This module will be removed in 0.20.
'''
def create_data():
    x,y = make_classification(n_samples=900,n_classes=2,n_features=50)
    np.savetxt('x.txt',x)
    np.savetxt('y.txt',y)

# -----------------------------------------------------
# 函数区
# -----------------------------------------------------
# ======================================================
# 计算壳向量
# ======================================================
def convex_hull(x,y):
    # 1、分别针对数据集A+和A-求壳向量集A+HV和A-HV,令AHV= A+HV∪ A-HV
    # 不同类别的数据分开,仅仅用于二分类问题，标签分别为0和1
    A_plus = x[y == 0]
    A_minus = x[y == 1]
    # print(sys._getframe().f_lineno)
    # 求不同类别数据集合的壳向量
    A_hv_plus = ConvexHull(A_plus)
    # print(sys._getframe().f_lineno)
    A_hv_minus = ConvexHull(A_minus)
    # print(sys._getframe().f_lineno)
    # 将壳向量合并
    A_hv = np.concatenate((A_plus[A_hv_plus.vertices, :], A_minus[A_hv_minus.vertices, :]), axis=0)
    A_hv_label = np.concatenate(((np.zeros(A_hv_plus.vertices.shape[0]) + 0),
                                 (np.zeros(A_hv_minus.vertices.shape[0]) + 1)), axis=0)
    return A_hv,A_hv_label

# -----------------------------------------------------
# 主程序
# -----------------------------------------------------
def convex_hull_svm(Batchs=20,Train_size = 0.05):
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
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=Train_size, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 计算hull向量集合
    A_hv, A_hv_label = convex_hull(train_x,train_y)

    # 2、将壳向量集AHV作为新的训练样本集,运行SVM算法得到支持向量集ASV,并由此构造最优分类决策函数
    ini_clf = svm.SVC()
    model2 = ini_clf.fit(A_hv, A_hv_label)
    result_y = ini_clf.predict(test_x)
    result = result_y - test_y
    print('=====================================')
    print('初始化分类器：')
    print('壳向量集作为新的训练样本集,正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('类别：%d 的支持向量数量为%d' % (0, ini_clf.n_support_[0]))
    print('类别：%d 的支持向量数量为%d' % (1, ini_clf.n_support_[1]))
    print('壳向量数量为%d' % (A_hv.shape[0]))


    # 3、开始增量学习
    batchs = Batchs #学习批次
    nums = np.floor(test_x.shape[0]/batchs).astype('int') # 进行切片运算时，必须是整数
    remainder = test_x.shape[0]%batchs
    results = np.zeros((batchs,9))
    print('=====================================')
    print('学习批次=\t',batchs)
    print('每批样本数量=\t', nums)
    print('余数=\t',remainder)
    print('=====================================')
    print('开始增量学习：')
    for i in np.arange(0, batchs):
        print('第%d次增量学习'%(i+1))
        train_size = 0.8
        # 用nums *  train_size 个新增样本进行训练，剩余进行测试

        A = np.concatenate(( test_x[i * nums : ((i + 1) * nums * train_size).astype('int'), :], A_hv), axis=0 )  # 令A= B∪AHV 作为新训练样本集
        A_label = np.concatenate([test_y[i * nums:((i + 1) * nums * train_size).astype('int')], A_hv_label])
        A_plus = A[A_label == 0]
        A_minus = A[A_label == 1]

        A_hv_plus = ConvexHull(A_plus)
        A_hv_minus = ConvexHull(A_minus)
        # 将壳向量合并
        A_hv = np.concatenate((A_plus[A_hv_plus.vertices, :], A_minus[A_hv_minus.vertices, :]), axis=0)
        A_hv_label = np.concatenate(((np.zeros(A_hv_plus.vertices.shape[0]) + 0),
                                     (np.zeros(A_hv_minus.vertices.shape[0]) + 1)), axis=0)
        t1 = time.time()
        clf = svm.SVC()
        model = clf.fit(A_hv, A_hv_label)
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums,:])
        t2= time.time()
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        print('类别：%d 的支持向量数量为%d' % (0, clf.n_support_[0]))
        print('类别：%d 的壳向量数量为%d' % (0, A_hv_plus.vertices.shape[0]))
        print('类别：%d 的支持向量数量为%d' % (1, clf.n_support_[1]))
        print('类别：%d 的壳向量数量为%d' % (1, A_hv_minus.vertices.shape[0]))
        print('耗时%f' % ((t2 - t1)))
        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i,:] = [i,
                        train_x.shape[0],# 初始训练样本数量
                        nums, #每轮循环新增样本数量
                        clf.n_support_[0],
                        A_hv_plus.vertices.shape[0],
                        clf.n_support_[1],
                        A_hv_minus.vertices.shape[0],
                        (np.sum(result == 0) / result.shape[0]),
                        (t2-t1)]

    np.savetxt('convex_hull_svm.txt',results)
    return

def normal_svm(Batchs=20,Train_size = 0.05):
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
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=Train_size, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 训练svm模型
    ini_clf = svm.SVC()
    model2 = ini_clf.fit(train_x, train_y)
    result_y = ini_clf.predict(test_x)
    result = result_y - test_y
    print('=====================================')
    print('初始化分类器：')
    print('壳向量集作为新的训练样本集,正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('类别：%d 的支持向量数量为%d' % (0, ini_clf.n_support_[0]))
    print('类别：%d 的支持向量数量为%d' % (1, ini_clf.n_support_[1]))

    # 3、开始增量学习
    batchs = Batchs  # 学习批次
    nums = np.floor(test_x.shape[0] / batchs).astype('int')  # 进行切片运算时，必须是整数
    remainder = test_x.shape[0] % batchs
    results = np.zeros((batchs, 7))
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

        batchs_train_x = np.vstack((train_x[:, :], test_x[0: ((i + 1) * nums * train_size).astype('int'), :]))
        batchs_train_y = np.concatenate((train_y[:], test_y[0: ((i + 1) * nums * train_size).astype('int')]),
                                        axis=0)
        t1 = time.time()
        clf = svm.SVC()
        model = clf.fit(batchs_train_x,
                        batchs_train_y
                        )
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums, :])
        t2 = time.time()
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        print('类别：%d 的支持向量数量为%d' % (0, clf.n_support_[0]))
        print('类别：%d 的支持向量数量为%d' % (1, clf.n_support_[1]))
        print('耗时%f' % ((t2-t1)))
        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i, :] = [i,
                         train_x.shape[0],# 初始训练样本数量
                         nums, #每轮循环新增样本数量
                         clf.n_support_[0],
                         clf.n_support_[1],
                         (np.sum(result == 0) / result.shape[0]),
                         (t2-t1)]

    np.savetxt('normal_svm.txt', results)
    return

def hull_svm(Batchs=20,ini_train_size = 0.05):
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
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=ini_train_size, random_state=0)

    # =======================================================
    # 三、训练分类模型
    # 1、训练初始svm
    ini_clf = svm.SVC()
    ini_model = ini_clf.fit(train_x, train_y)
    result_y = ini_clf.predict(test_x)
    result = result_y - test_y
    sv = ini_model.support_vectors_
    sv_label = train_y[ini_model.support_]
    print('=====================================')
    print('初始化分类器：')
    print('初始化分类器的正确率：%f' % (np.sum(result == 0) / result.shape[0]))
    print('支持向量数量为%d' % (sv.shape[0]))


    # 3、开始增量学习
    batchs = Batchs #学习批次
    nums = np.floor(test_x.shape[0]/batchs).astype('int') # 进行切片运算时，必须是整数
    remainder = test_x.shape[0]%batchs
    results = np.zeros((batchs,7))
    print('=====================================')
    print('学习批次=\t',batchs)
    print('每批样本数量=\t', nums)
    print('余数=\t',remainder)
    print('=====================================')
    print('开始增量学习：')
    for i in np.arange(0, batchs):
        print('第%d次增量学习'%(i+1))
        train_size = 0.8
        # 用nums *  train_size 个新增样本进行训练，剩余进行测试

        train_data = np.concatenate(( test_x[i * nums : ((i + 1) * nums * train_size).astype('int'), :], sv), axis=0 )  # 令A= B∪AHV 作为新训练样本集
        train_data_label = np.concatenate(( test_y[i * nums : ((i + 1) * nums * train_size).astype('int')], sv_label),
                                          axis=0 )

        t1 = time.time()
        clf = svm.SVC()
        model = clf.fit(train_data, train_data_label)
        # 用剩余的部分进行测试
        result_y = clf.predict(test_x[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums,:])
        t2= time.time()
        result = result_y - test_y[((i + 1) * nums * train_size).astype('int'):(i + 1) * nums]
        print('正确率：%f' % (np.sum(result == 0) / result.shape[0]))
        print('类别：%d 的支持向量数量为%d' % (0, clf.n_support_[0]))
        print('类别：%d 的支持向量数量为%d' % (1, clf.n_support_[1]))
        print('耗时%f' % ((t2 - t1)))
        sv = model.support_vectors_
        sv_label = train_data_label[model.support_]
        # 自动判断是否以科学计数法输出,'{:g}'.format(i),没有效果
        results[i,:] = [i,
                        train_x.shape[0],# 初始训练样本数量
                        nums, #每轮循环新增样本数量
                        clf.n_support_[0],
                        clf.n_support_[1],
                        (np.sum(result == 0) / result.shape[0]),
                        (t2-t1)]

    np.savetxt('hull_svm.txt',results)
    return

if __name__ == '__main__':
    create_data()
    batchs = 50
    hull_svm(Batchs=batchs)
    convex_hull_svm(Batchs=batchs)
    normal_svm(Batchs=batchs)
    hull_svm(Batchs=batchs)
    # 绘图
    # os.chdir('D:\\20_同步文件\\pyStudy\\Apple_essence')
    results_hull_svm = np.loadtxt('hull_svm.txt')
    results_convex_hull_svm = np.loadtxt('convex_hull_svm.txt')
    results_normal_svm = np.loadtxt('normal_svm.txt')

    plt.figure(1)
    # 图1
    plt.subplot(121)
    plt.plot(results_hull_svm[:, 0], results_hull_svm[:, -2] *100, '-*',label='SV-SVM')
    plt.plot(results_convex_hull_svm[:, 0], results_convex_hull_svm[:, -2]* 100, '-+',label='CCH-SVM')
    plt.plot(results_normal_svm[:, 0], results_normal_svm[:, -2] * 100, '-D', label='SVM')
    plt.ylim((40, 100))
    # plt.title('Classification accuracies of the normal svm and hull svm')
    plt.xlabel('learn nums')
    plt.ylabel('Classification accuracy(%)')
    plt.legend(loc='best')
    # 图2
    plt.subplot(122)
    plt.plot(results_hull_svm[:, 0], results_hull_svm[:, -1] *100, '-*',label='SV-SVM')
    plt.plot(results_convex_hull_svm[:, 0], results_convex_hull_svm[:, -1]* 100, '-+',label='CCH-SVM')
    plt.plot(results_normal_svm[:, 0], results_normal_svm[:, -1] * 100, '-D', label='SVM')

    plt.xlabel('learn nums')
    plt.ylabel('raining time(s)')
    plt.legend(loc='best')
    plt.title('training time of the normal svm and hull svm')
    plt.show()