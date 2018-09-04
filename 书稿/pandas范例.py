# -*- coding: utf-8 -*-
'''
https://www.cnblogs.com/chaosimple/p/4153083.html
'''
import pandas as pd
import numpy as np

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

#%% 通过位置选择
# 通过传递数值进行位置选择（选择的是行）
df.iloc[3]
# 获取特定的值
df.iloc[3][0]

# 通过数值进行切片，与numpy/python中的情况类似
df.iloc[0:2,0:1]

#%% 布尔索引
# 使用一个单独列的值来选择数据：
df[df.A >0.5]
df[df.A >0.8]
# 使用where操作来选择数据
df[df > 0]

#%% 设置
# 通过位置设置新的值
df.iat[0,1] =99
df.iloc[0][1] = 98

#%% 缺失值处理
# 去掉包含缺失值的行：
df.dropna(how='any')

# 对缺失值进行填充：
df.fillna(values=5)




#%%
# 行的选取
rows = df[0:3]

# 列的选取
#选择一个单独的列，这将会返回一个Series，等同于df.A：
cols = df[['A', 'B', 'C']]

# 获取一个标量
df.loc[0,'A']

# 块的选取
df.loc['20130102':'20130104',['A','B']]

# 根据条件过滤行
df[(df.index >= '2013-01-01') & (df.index <= '2013-01-03')]
df[df['A'] > 0]

# 查看数据
df.head(5) # 头
df.tail(4) # 尾

# 显示索引、列和底层的numpy数据：
df.index
df.columns
df.values

# describe()函数对于数据的快速统计汇总
df.describe()

# 按值进行排序
df.sort_values(by='A')