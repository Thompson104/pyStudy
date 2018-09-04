print(__doc__)
'''
超参的调整与优化
'''
import matplotlib.pyplot as plt
import pandas as pd
import time
# 网格搜索的相关包
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,accuracy_score
from sklearn.pipeline import Pipeline
# 学习模型
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
if __name__ == '__main__':
    # =========================================================
    '''
    管道机制在机器学习算法中得以应用的根源在于，参数集在新数据集（比如测试集）上的重复使用。
    管道机制实现了对全部步骤的流式化封装和管理（streaming workflows with pipelines）。
    Pipeline按顺序构建一系列转换和一个模型，最后的一步是模型。Pipeline中间的步骤必须是转换过程，
    它们必须包含fit和transform方法。最后一步模型只要有fit方法。
    Pipeline的目的是能组合好几个步骤，当设置不同参数的时候，可以在一起做交叉验证。
    可以通过【pipeline的名称+ “__” + 参数名称】(注意是两个下划线)的方式设置多个步骤的参数。
    '''
    pipeline = Pipeline([
        ('vect',TfidfVectorizer(stop_words='english')),
        ('clf',LogisticRegression())
    ])
    parameters = {
        'vect__max_df': (0.25, 0.5, 0.75),
        'vect__stop_words': ('english', None),
        'vect__max_features': (2500, 5000, 10000, None),
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__use_idf': (True, False),
        'vect__norm': ('l1', 'l2'),
        'clf__penalty': ('l1', 'l2'),
        'clf__C': (0.01, 0.1    , 1, 10),
    }
    '''
    GridSearchCV()函数的参数有待评估模型pipeline，超参数词典parameters和效果评价指
    标scoring。n_jobs是指并发进程最大数量，设置为-1表示使用所有CPU核心进程。在Python3.4
    中，可以写一个Python的脚本，让fit()函数可以在main()函数里调用，也可以在Python自带命令
    行,IPython命令行和IPython Notebook运行。经过网格计算后的超参数在训练集中取得了很好的效
    果。
    '''
    startTime = time.time()
    grid_search = GridSearchCV(pipeline,parameters,n_jobs=2,verbose=1,scoring='accuracy',cv=3)

    df = pd.read_csv(r'../input/SMSSpamCollection',delimiter='\t',names=['label','message'])
    x = df['message']
    y = df['label']
    # 对标签进行编码，预处理
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)

    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75)

    grid_search.fit(x_train,y_train)
    best_param = grid_search.best_estimator_.get_params()
    # 输出参数
    for param in sorted(best_param.keys()):
        print('\t%s: %r' % (param,best_param[param]))
    predictions = grid_search.predict(x_test)
    print('准确率：', accuracy_score(y_test, predictions))
    print('精确率：', precision_score(y_test, predictions))
    print('召回率：', recall_score(y_test, predictions))
    endTime = time.time()
    print('耗时：%0.3f分钟'% ( (endTime - startTime )/60 ) )


