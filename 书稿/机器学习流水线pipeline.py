# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:03:14 2018

@author: TIM
"""
from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,f_classif,f_regression
from sklearn.pipeline import Pipeline

#%% 生成模拟数据
X,y = samples_generator.make_classification(n_informative=4,
                                            n_features=20,
                                            n_redundant=2,
                                            random_state=0)

#%% 选择K个最好的特征
selector_k_best = SelectKBest(f_classif,k=10)

#%% 用随机森林分类器分类数据
classifier = RandomForestClassifier(n_estimators=50,max_depth=4)

#%% 构建流水线
pipeline_classifier = Pipeline(steps=[('特征选择器',selector_k_best),('随机森林分类器',classifier)])
# 设置参数
print(pipeline_classifier.get_params().keys())
pipeline_classifier.set_params(特征选择器__k=6,随机森林分类器__n_estimators=100)

#%% 训练分类器
pipeline_classifier.fit(X,y)

prediction = pipeline_classifier.predict(X)
print('\n 预测结果：\n',prediction)

#%% 评价分类器的性能
print('\nScore:\n',pipeline_classifier.score(X,y))

#%% 查看那些特征被选中了
features_status = pipeline_classifier.named_steps['特征选择器'].get_support()
selected_features = []
for count,item in enumerate(features_status):
    if item:
        selected_features.append(count)
print('\n 被选择的特征（0-indexed）：\n',','.join([str(x) for x in selected_features]))
