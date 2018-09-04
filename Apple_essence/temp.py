import numpy as np
import pandas as pd
from sklearn import decomposition as dp
from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  #归一化
from sklearn.decomposition import PCA,KernelPCA
from sklearn.datasets import  make_classification
from mpl_toolkits.mplot3d import Axes3D
from scipy import  io as spi
import time as tm
import os
from sklearn import datasets
import pickle

iris = datasets.load_iris()
x = iris.data
y = iris.target

clf = svm.SVC()
clf.fit(x,y)
clf_save = pickle.dumps(clf)

f = open('model_svm.txt','w')
f.write(clf_save)
f.close()

