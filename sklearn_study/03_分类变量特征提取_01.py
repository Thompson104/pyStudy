# -*- coding: utf-8 -*-
"""
分类变量特征提取
"""
from sklearn.feature_extraction import DictVectorizer
# 独热编码
onehot_encoder = DictVectorizer()
instances = [{'city':'南京'},{'city':'苏州'},{'city':'上海'}]
print(onehot_encoder.fit_transform(instances).toarray())

