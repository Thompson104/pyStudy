# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt

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