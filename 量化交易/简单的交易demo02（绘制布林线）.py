# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:36:53 2018

@author: smart
"""

import numpy as np
import pandas as pd
import talib as ta
import tushare as ts

sh = ts.get_hist_data('sh').sort_index()

close = sh[['close','volume']]

ma10 = ta.MA(close['close'],10)
close['ma10'] = ma10

#close[['close','ma10']].plot()

# 布林线
upper,middle,lower  = ta.BBANDS( sh['close'].values,
                                 timeperiod = 20,
                                 nbdevup = 2,
                                 nbdevdn = 2,
                                 matype = 0)
close.loc[:,'upper'] = upper
close.loc[:,'middle'] = middle
close.loc[:,'lower'] = lower
# 绘制布林带
close[['close','middle','upper','lower']].plot(figsize=(10,5))


