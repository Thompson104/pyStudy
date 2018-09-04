# -*- coding: utf-8 -*-
"""
沪深300指数股交易信息下载，并插入数据库
Created on Tue May 30 01:46:33 2017
@author: TIM
"""
import tushare as ts
import datetime as dt
import pandas as pd
import numpy as np
import pymysql

db_host = 'localhost'
db_user = 'sec_user'
db_pass = 'sunshine721226'
db_name = 'securities_master'
conn = pymysql.connect(
    host=db_host,
    user=db_user,
    password=db_pass,
    database=db_name,
    use_unicode=True, charset="utf8")


# ======================================================
def obtain_list_of_db_tickers():
    """
    从数据库中获取沪深300指数股的信息.
    """
    with conn:
        cur = conn.cursor()
        cur.execute(u"SELECT id, ticker,code FROM symbol where currency ='rmb' ")
        data = cur.fetchall()
        return [(d[0], d[1], d[2]) for d in data]


def get_daily_historic_data_tushare(code,
                                    start_date='2010-01-01',
                                    end_date=dt.datetime.now().strftime('%Y-%m-%d'),
                                    ktype='D'):
    """
    从tushare获取特定股票的历史价格
    """
    try:
        f_data = ts.get_hist_data(code, start_date, end_date, ktype)
        prices = []
        for i in np.arange(f_data.shape[0]):
            prices.append(
                (f_data.index[i],  # f_data['date'][i],
                 f_data['open'][i], f_data['high'][i], f_data['low'][i],
                 f_data['close'][i], f_data['volume'][i], f_data['ma5'][i])
            )
    except Exception as e:
        print("无法获取代码为[%s]的股票历史数据：%s" % (code, e))
    return prices


def insert_daily_data_into_db(data_vendor_id, symbol_id, daily_data):
    now = dt.datetime.now().strftime('%Y-%m-%d')
    #
    column_str = """data_vendor_id, symbol_id, price_date, created_date, 
                 last_updated_date, open_price, high_price, low_price, 
                 close_price, volume, adj_close_price"""
    insert_str = ("%s, " * 11)[:-2]
    final_str = "INSERT INTO daily_price (%s) VALUES (%s)" % \
                (column_str, "%s,%s,'%s','%s','%s',%s,%s,%s,%s,%s,%s")

    daily_data1 = [
        (data_vendor_id, symbol_id, d[0], now,
         now, d[1], d[2], d[3], d[4], d[5], d[6]) for d in daily_data
    ]

    with conn:
        try:
            # temp = final_str % daily_data1[1]
            cur = conn.cursor()
            #必须通过循环，单行执行insert语句
            for row in daily_data1:
                temp = final_str % row
                cur.execute(temp)
            # 批量插入会报错，float64 translate error
            #cur.executemany(final_str, daily_data1)

        except Exception as e:
            print("错误信息：%s" % (e))
        return

if __name__ == "__main__":

    # Loop over the tickers and insert the daily historical
    # data into the database
    tickers = obtain_list_of_db_tickers()
    lentickers = len(tickers)
    for i, t in enumerate(tickers):
        print(
            "添加数据中 %s: 第 %s 共 %s" %
            (t[2], i + 1, lentickers)
        )
        yf_data = get_daily_historic_data_tushare(t[2])
        insert_daily_data_into_db('1', t[0], yf_data)
    print("成功地将tushare中沪深300股票的历史交易数据加载到数据库.")
