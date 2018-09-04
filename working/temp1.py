# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:25:25 2017

@author: TIM
"""
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
from matplotlib.dates import MonthLocator
from matplotlib.finance import quotes_historical_yahoo_ohlc
from matplotlib.finance import candlestick_ohlc
import sys
from datetime import date
today = date.today()

start = (today.year -1,today.month,today.day)

Alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter('%b %Y')

symbol = 'DISH'

quotes = quotes_historical_yahoo_ohlc(symbol,start,today)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.xaxis.set_major_locator(months)

candlestick_ohlc(ax,quotes)
fig.autofmt_xdate()
plt.show()