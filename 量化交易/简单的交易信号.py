import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import tushare as ts

hs300 = ts.get_hist_data('hs300').sort_index()

#hs300[['close','ma10','ma20']].plot()

spread = 3 
hs300['short'] = np.round( hs300['close'].rolling(10).mean() ,2)
hs300['long'] = np.round( hs300['close'].rolling(40).mean() ,2)
hs300['short-long'] = hs300['short'] - hs300['long']

#hs300[['close','long','short']].plot()

# 多空信号提取:0空仓，1多头，-1空头
hs300['signal'] = np.where(hs300['short-long'] > spread,
     1,# 多仓
     0)
hs300['signal'] = np.where(hs300['short-long'] < spread,
     -1, # 空仓
     hs300['signal'])# 保存原来信号

hs300['signal'].value_counts()

hs300['signal'].plot()
# 今天的收盘价除以昨天的收盘价，shift（1）数据平移，这样才能没有【未来信号】！！！
# ！！！！避免用明天的数据！！！！
# 添加market列用对数形式记录走势
# 归一化:用log归一化的好处：有正负号（大于1为正直，小于1为负值，正好体现市场的涨跌）
hs300['market'] = np.log( hs300['close'] / hs300['close'].shift(1) ) 
#plt.scatter(range(len(hs300['market'])),hs300['market'])
# 如果除以最大值则反应不出市场的涨跌
hs300['market1'] = (hs300['close'] / hs300['close'].shift(1) ) / np.max( hs300['close'] / hs300['close'].shift(1) )
# 添加straegy列用于记录相对于走势的资金曲线，
# 相对于走势的资金 = 昨天的多空信号 * 今天市场走势 
hs300['straegy'] = hs300['signal'].shift(1) * hs300['market']

# cumsum进行累加，显示市场总体走向
hs300[['market','straegy']].cumsum().apply(np.exp).plot()
hs300[['market','straegy']].cumsum().plot()
