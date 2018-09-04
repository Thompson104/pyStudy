# -*- coding: utf-8 -*-
"""
词库表示法,语料库
一批文档的集合称为文集（corpus）
# 文档向量的降维： 停用词、词根还原、词形还原
# tf-idf调整词频
"""
from sklearn.metrics.pairwise import euclidean_distances #欧氏距离
from sklearn.feature_extraction.text import CountVectorizer
#===================================================
corpus = [
            'UNC played Duke in basketball',
            'Duke lost the basketball game',
            'I ate a sandwich'
        ]
'''
CountVectorizer旨在通过计数来将一个文档转换为向量。
当不存在先验字典时，Countvectorizer作为Estimator提取词汇进行训练，
并生成一个CountVectorizerModel用于存储相应的词汇向量空间。
该模型产生文档关于词语的稀疏表示，其表示可以传递给其他算法，例如LDA。
'''
# CountVectorizer类通过正则表达式用空格分割句子，然后抽取长
#度大于等于2的字母序列。
vectorizer = CountVectorizer()
vectorizer.set_params()
result = vectorizer.fit_transform(corpus)
print(result.todense())
# 输出
print(vectorizer.vocabulary_)

# 计算文档间的距离
counts = result.todense()
for x,y in [[0,1],[0,2],[1,2]]:
    dist = euclidean_distances(counts[x],counts[y])
    print('文档{}与文档{}的距离{}'.format(x,y,dist))
    

    