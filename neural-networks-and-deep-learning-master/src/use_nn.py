# -*- coding: utf-8 -*-
"""
程序说明：
作者：刘军
"""
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import network
import mnist_loader

training_data , validation_data , test_data = mnist_loader.load_data_wrapper()
# training_data = list(training_data)
# validation_data = list(validation_data)
# test_data = list(test_data)
net = network.Network([784,30,10])
net.descript()
net.SGD( training_data[0:-1],10,1,0.3,test_data=test_data[0:10])

