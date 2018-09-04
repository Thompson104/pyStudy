# -*- coding:,utf-8,-*-
import jieba as jb
import numpy as np
import copy
# 文本分类
ftest1fn = 'mobile2.txt'
ftest2fn = 'war1.txt'
sampfn = 'war2.txt'
# 余弦相似度
def get_cossimi(x,y):
    myx = np.array(x)
    myy = np.array(y)
    cos1 = np.sum(myx * myy)
    cos21 = np.sqrt(np.sum(myx*myx))
    cos22 = np.sqrt(np.sum(myy*myy))
    return  cos1/float(cos21 * cos22)
# 欧氏距离相似度
def get_EuclideanSimi(x,y):
    myx = np.array(x)
    myy = np.array(y)
    return np.sqrt(np.sum((myx-myy)*(myx-myy)))
# 马氏距离
def get_MahalanobisSimi(x,y):
    myx = np.array(x)
    myy = np.array(y)
    meanX = np.mean(myx)
    meanY = np.mean(myy)
    covx = np.cov(myx,myy)


if __name__ == '__main__':
    print()
    print('数据加载中...')
    print('工作中.......')
    f1 = open(sampfn,encoding='utf-8')
    try:
        f1_text = f1.read()
    finally:
        f1.close()

    f1_seg_list = jb.cut(f1_text,cut_all=False)

    # 第一个待测试数据
    ftest1 = open(ftest1fn,encoding='utf-8')
    try:
        ftest1_text = ftest1.read()
    finally:
        ftest1.close()
    ftest1_seg_list = jb.cut(ftest1_text,cut_all=False)

    # 第二个待测试数据
    ftest2 = open(ftest2fn,encoding='utf-8')
    try:
        ftest2_text = ftest2.read()
    finally:
        ftest2.close()
    ftest2_seg_list = jb.cut(ftest2_text, cut_all=False)

    # 读取停用词
    f_stop = open('stopwords.txt',encoding='utf-8')
    try:
        f_stop_text = f_stop.read()
    finally:
        f_stop.close()
        f_stop_seg_list = f_stop_text.split('\n')

    # 构造样本词的字典
    test_words = {}
    all_words = {}
    for myword in f1_seg_list:
        print('',end='.')
        if not(myword.strip() in f_stop_seg_list):
            # setdefault 如果键不已经存在于字典中，将会添加键并将值设为默认值
            test_words.setdefault(myword,0)
            all_words.setdefault(myword,0)
            all_words[myword] += 1

    # 读取测试文本，已样本文本的字典为基础，进行测试文本的字频统计
    # 只有这样“字词”才一一对应
    mytest1_words = copy.deepcopy(test_words)
    print('\n')
    for myword in ftest1_seg_list:
        print('',end='.')
        if not (myword.strip() in f_stop_seg_list):
            if myword in mytest1_words:
                mytest1_words[myword] += 1

    mytest2_words = copy.deepcopy(test_words)
    print('\n')
    for myword in ftest2_seg_list:
        print('', end='.')
        if not (myword.strip() in f_stop_seg_list):
            if myword in mytest1_words:
                mytest2_words[myword] += 1
    # 构造向量，list列表数据类型
    sampdata = []
    test1data = []
    test2data = []
    for key in all_words.keys():
        # 取出每个键对应的值
        sampdata.append(all_words[key])
        test1data.append(mytest1_words[key])
        test2data.append(mytest2_words[key])
    # 计算余弦相似度
    test1simi = get_cossimi(sampdata,test1data)
    test2simi = get_cossimi(sampdata, test2data)
    print('\n')
    print(u'%s与样本[%s]的余弦相似度：%f\n'%(ftest1fn,sampfn,test1simi))
    print(u'%s与样本[%s]的余弦相似度：%f' % (ftest2fn, sampfn, test2simi))
    print('\n')
    temp1simi = get_EuclideanSimi(sampdata,test1data)
    temp2simi = get_EuclideanSimi(sampdata, test2data)
    print(u'%s与样本[%s]的欧氏距离相似度：%f\n' % (ftest1fn, sampfn, temp1simi))
    print(u'%s与样本[%s]的欧氏距离相似度：%f' % (ftest2fn, sampfn, temp2simi))