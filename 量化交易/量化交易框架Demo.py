# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 21:10:50 2017

@author: TIM
"""

def init(context):
    g.security = '002043.XSHE'# 存入兔宝宝的股票代码
    return

def handle_data(context,data):
    print(g.secutity)
    return
    
if __name__ == '__main__':
    init(None)
    handle_data(None,None)