import  numpy as np
import scipy.io as spi


def load_Laman():
    '''
    10种苹果香精的拉曼（XJlaman_4_1.mat）和离子迁移谱（XJlizi_4_1.mat）数据，
    每种香精只购买一个批次，分别于不同时间段采集了30个数据，总共就是10*30=300个拉曼数据和300个离子迁移谱数据，
    数据格式为matlab格式，需要用matlab软件打开。
    拉曼数据的列数为300列，1-30列为A香精的数据，31-60为B香精数据，依次类推，
    离子迁移谱数据分别与拉曼数据一一对应，也是300列。拉曼数据的行数为2090，代表一张拉曼谱图采集了2090个点，
    同理离子迁移谱的6000行代表采集了6000个点。
    '''
    '''加载拉曼谱图数据'''
    raw_data = spi.loadmat('../input/XJlaman_4_1.mat')
    x = raw_data['lamanxiangjing']
    y = np.zeros((x.T.shape[0]))
    for i in np.arange(0,30):
        y[i*30:i*30+30,]= i
    return x.T,y


def load_IMS():
    '''
    10种苹果香精的拉曼（XJlaman_4_1.mat）和离子迁移谱（XJlizi_4_1.mat）数据，
    每种香精只购买一个批次，分别于不同时间段采集了30个数据，总共就是10*30=300个拉曼数据和300个离子迁移谱数据，
    数据格式为matlab格式，需要用matlab软件打开。
    拉曼数据的列数为300列，1-30列为A香精的数据，31-60为B香精数据，依次类推，
    离子迁移谱数据分别与拉曼数据一一对应，也是300列。拉曼数据的行数为2090，代表一张拉曼谱图采集了2090个点，
    同理离子迁移谱的6000行代表采集了6000个点。
    '''
    '''加载离子迁移谱数据'''
    raw_data = spi.loadmat('../input/XJlizi_4_1_1.mat')
    x = raw_data['XJlizi_4_1_1']
    y = np.zeros((x.T.shape[0]))
    for i in np.arange(0,30):
        y[i*30:i*30+30,]= i
    return x.T,y