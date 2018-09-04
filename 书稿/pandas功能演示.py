


# 导入 Numpy 包，并指定别名为 np
import numpy as np
# 导入 pandas 包，并指定别名为 pd
import pandas as pd

#%% 创建Pandas对象
# 创建一个Series
s = pd.Series([1,3,5,7,6,8])
print( s )
# 获取 Series 的索引
s_index = s.index
print(s_index)

# 创建一个DataFrame
df_value = np.random.randn(4,3)
df_colums = np.array(['one','two','three'])
df = pd.DataFrame(data=df_value,columns=df_colums)

#%% 查看数据
# 查看DataFrame的行、列和值
print(df.index) #查看行
print(df.columns) #查看列
print(df.values) #查看元素

# 查看前几条数据
df.head()
# 查看后几条数据
df.tail()

# 使用 describe() 函数对于数据的快速统计汇总
print(df.describe())

#%% 选择数据
# 根据列名选择数据,注意返回值类型为DataFrame
df[['one','two']]
# 选择多行,注意返回值类型为DataFrame
df[0:3]
# 使用标签选取数据
'''
loc——通过行标签索引行数据
'''
# 选取第2行的‘one’列的数据
df.loc[1,'one']  
## 获取多行数据,
df.loc[1:3] # 需要注意的是，dataframe的索引[1:3]包含1,2,3
'''
iloc——通过行号获取行数据
'''
df.iloc[:,[1]] #在iloc中只能使用行或列号
df.iloc[0:] # 通过行号可以索引多行