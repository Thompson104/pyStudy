#
'''
现实中有很多问题不只是分成两类，许多问题都需要分成多个类，成为多类分类问题（Multi-class
classification）。比如听到一首歌的样曲之后，可以将其归入某一种音乐风格。这类风格就有许多
种。scikit-learn用one-vs.-all或one-vs.-the-rest方法实现多类分类，就是把多类中的每个类都作为二
元分类处理。分类器预测样本不同类型，将具有最大置信水平的类型作为样本类
型。
'''
# ================================
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# ================================
if __name__=='__main__':
    print(__doc__)