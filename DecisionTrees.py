# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    # 字典
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    
    shannonEnt = 0.0
    # 计算数据集的信息熵，由每个标签数据的数量除以数据集数据的数量等
    for key in labelCounts:
        prob = float( labelCounts[key] / numEntries )
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt
    
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']
            ]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):
    """
    按照给定特征划分数据集
    dataSet:数据集
    axis:特征
    value：需要返回特征的取值
    return：返回指定数轴axis等于特定值value的dataset的子集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 跳过axis所在的数据项
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    # 取数据集的第一行的数据的个数，即特征的数量
    # 如果使用numpy则，可以通过narray的shape属性来获取
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestinfoGain = 0.0
    bestFeatures = -1
    
    # 对数据集中的每一个特征进行循环
    for i in range(numFeatures):
        # 取数据集中的第i列数据，即第i个特征的所有值
        featList = [example[i] for example in dataSet]
        # 集合的元素是唯一的set([1, 1, 1, 0, 0]) == {0, 1}
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 第i各特征的每个取值计算信息熵，并累加
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntropy - newEntropy
        # 记录是否是最大增益特征
        if (infoGain > bestinfoGain ):
            bestinfoGain = infoGain
            bestFeatures = i
    return bestFeatures

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedclassCount = sorted(classCount.items(),key=operator.itemgetter[1],reversed=True)
    return sortedclassCount

def CreateTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同，这停止划分，退出
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果只有一个特征，则遍历该特征，返回出现次数最多的
    if len(dataSet[0] )== 1:
        return majorityCnt(classList)
    # 构造根节点
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    mytree = {bestFeatLabel:{}}
    # 将最佳标签删除
    operator.delitem(labels,bestFeat)
    #operator.del(labels(bestFeat))
    # 最佳标签的取值list
    featValues = [example[bestFeat] for example in dataSet]
    # 去除重复值
    uniqueVals = set(featValues)    
    for value in uniqueVals:
        subLabels = labels[:]
        mytree[bestFeatLabel][value]= CreateTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return mytree


d,a = createDataSet()

xx =CreateTree(d,a)
print(xx)