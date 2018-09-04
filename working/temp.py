import  numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.metrics as metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split,cross_val_score

# 加载数据
iris = datasets.load_iris()

# 构造样品数据

train_data,test_data,train_label,test_label_expected = \
                        train_test_split(iris.data,iris.target,test_size=0.51,random_state=0)


