# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:31:17 2017

@author: TIM
"""

import tushare as ts
import pandas as pd
import datetime as dt
import numpy as np
import pymysql

def obtain_hs300():    
    # 记录当期时间
    now = dt.datetime.now()
    
    # 获取沪深300清单
    hs300s = ts.get_hs300s()
    
    # 获取所有股票的基本信息
    stock_basics = ts.get_stock_basics()
    
    hs300s = pd.merge(hs300s,stock_basics,how='inner')
    
    symbols = []
    
    for i in np.arange(hs300s.shape[0]):
        symbols.append(
            (
                hs300s.code[i],         # code
                hs300s.name[i],         # ticker
                'stock',                # instrument
                hs300s.name[i],         # Name
                hs300s.industry[i],     # Sector
                'RMB',                  # currency
                now, 
                now
            ) 
        )
        
    return symbols
                
def insert_hs300_symbols(symbols):
    
    # 连接mysql的参数
    db_host = 'localhost'
    db_user = 'sec_user'
    db_pass = 'sunshine721226'
    db_name = 'securities_master'
    
    # 连接mysql
    # 若没有 use_unicode=True, charset="utf8" 那么就会发生如题错误:
    #  UnicodeEncodeError: 'latin-1' codec can't encode characters 
    # in position 12-15: ordinal not in range(256)
    conn = pymysql.connect(
            host=db_host,
            user=db_user,
            password=db_pass,
            database= db_name,
            use_unicode=True, charset="utf8")
    
    # 生成insert语句
    column_str = """code,ticker, instrument, name, sector, 
                 currency, created_date, last_updated_date
                 """
    insert_str = ("%s, " * 8)[:-2]
    final_str = "INSERT INTO symbol (%s) VALUES (%s)" % \
             (column_str, insert_str)
    #print(final_str)


    with conn:
        cur = conn.cursor()
        # 注释，防止重复插入数据
        #cur.executemany(final_str, symbols)


