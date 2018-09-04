# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 23:11:24 2018

@author: smart
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import datetime
import scipy.misc
import scipy.io

#%% 便利函数定义
def _conv_layer(data,weights,bias):
    conv = tf.nn.conv2d(data,
                        tf.constant(weights),
                        strides=(1,1,1,1),
                        padding='SAME')
    return tf.nn.bias_add(conv,bias)
def _pool_layer(data):
    return tf.nn.max_pool(data,ksize=(1,2,2,1),strides=(1,2,2,1),padding='SAME')
def preprocess(image,mean_pixel):
    return image - mean_pixel
def unprocess(image,mean_pixel):
    return image + mean_pixel
def imgread(path):
    import imageio
    return imageio.imread(path).astype(np.float32)
#    return scipy.misc.imread(path).astype(np.uint8)
def imsave(path,img):
    img = np.clip(img,0,255).astype(np.uint8)
    return scipy.misc.imsave(path,img)

#%% 设置卷积网络结构
# 计算各层的前向传播，并记录
def net(data_path,input_image):
    # set数据类型
    layers = (
        'conv1_1','relu1_1','conv1_2','relu1_2','pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',  'conv3_3','relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1','conv4_2', 'relu4_2','conv4_3', 'relu4_3','conv4_4', 'relu4_4','pool4',
        'conv5_1', 'relu5_1','conv5_2', 'relu5_2','conv5_3', 'relu5_3','conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean,axis=(0,1))
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i,name in enumerate(layers):
        print(i,name)
        kind = name[:4]
        if kind == 'conv':
            kernels,bias = weights[i][0][0][0][0]
            #交换列,模型中权重的排列与tensorflow中的约定不同
            kernels = np.transpose(kernels,(1,0,2,3))
            bias = bias.reshape(-1)
            current = _conv_layer(current,kernels,bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        # net字典：保存各层的结果
        net[name] = current
    assert len(net) == len(layers)
    return net,mean_pixel,layers

if __name__ == '__main__':
    cwd = os.getcwd()
    VGG_PATH = cwd + '\imagenet-vgg-verydeep-19.mat'
    IMG_PATH = cwd + '\cat.jpg'
    input_image = imgread(IMG_PATH)
    shape = (1,input_image.shape[0],input_image.shape[1],input_image.shape[2])
    with tf.Session() as sess:
        image = tf.placeholder('float',shape=shape)
        #
        nets,mean_pixel,all_layers = net(VGG_PATH,image)
        input_image_pre = np.array([preprocess(input_image,mean_pixel)])
        layers = all_layers
        for i ,layer in enumerate(layers):
            print('[%d of %d] %s' %(i+1,len(layers),layer))
            features = nets[layer].eval(feed_dict={image:input_image_pre})
            print('Type of features is ',type(features))
            print('Shape of features is %s' %(features.shape,))
            # 显示中间特征图，将矩阵显示为图像
            if 1:
                plt.figure(i+1,figsize=(20,15))
                plt.matshow(features[0,:,:,0],cmap=plt.cm.gray,fignum=i+1)
                plt.title('' + layer)
                plt.colorbar()
                plt.show()
            
