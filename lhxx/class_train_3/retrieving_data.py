import pandas as pd
import pymysql as mdb

if __name__ == "__main__":
    # Connect to the MySQL instance
    db_host = 'localhost'
    db_user = 'sec_user'
    db_pass = 'sunshine721226'
    db_name = 'securities_master'
    conn = mdb.connect(
        host=db_host,
        user=db_user,
        password=db_pass,
        database=db_name,
        use_unicode=True, charset="utf8")
    # 查询‘吉祥航空’的股票信息
    sql = """SELECT dp.price_date, dp.adj_close_price
                 FROM symbol AS sym
                 INNER JOIN daily_price AS dp
                 ON dp.symbol_id = sym.id
                 WHERE sym.ticker = '吉祥航空'
                 ORDER BY dp.price_date ASC;"""

    # Create a pandas dataframe from the SQL query
    goog = pd.read_sql_query(sql, con=conn, index_col='price_date')

    # Output the dataframe tail
    print(goog.tail())