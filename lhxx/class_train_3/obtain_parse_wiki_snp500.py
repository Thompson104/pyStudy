import datetime
import requests
import bs4

def obtain_parse_wiki_snp500():
    """
    通过requests和BeautifulSoup下载、解析S&P500股票的名称等信息
    返回一个tuples 的list
    """
    # 记录当期时间，
    now = datetime.datetime.utcnow()   
    # 使用requests和 BeautifulSoup下载S&P500股票
    reponse = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    
    soup = bs4.BeautifulSoup(reponse.text)
    # 选择第一个table，
    # 忽略表头，从row（[0:]）
    symbolslist = soup.select('table')[0].select('tr')[1:]
    # 从表格的每一行获取股票信息
    symbols = []
    for i, symbol in enumerate(symbolslist):
        tds = symbol.select('td')
        # 添加tuples类型
        symbols.append(
                            (
                                tds[0].select('a')[0].text,     
                                'stock',
                                tds[1].select('a')[0].text,     # name
                                tds[3].text,                    # Sector
                                'USD',
                                now,
                                now
                            )
                        )
    return symbols
