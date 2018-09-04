# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 23:11:24 2018

@author: smart
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import datetime
#%% 加载数据
mnist = input_data.read_data_sets('mnist_data',one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

#%% 设置卷积网络结构
# 网络结构：{conv1（3 * 3 * 64） + pool1（2 * 2）} 
#         + {conv2（3 * 3 * 128） + pool2（2 * 2）} 
#         + {fc1(1024)}
#         + {fc2(10)}
n_input = 784
n_output = 10
stddev = 0.1
weights = {
        'wc1':tf.Variable(tf.random_normal([3,3,1,64],stddev=stddev)),
        'wc2':tf.Variable(tf.random_normal([3,3,64,128],stddev=stddev)),
        'wd1':tf.Variable(tf.random_normal([7*7*128,1024],stddev=stddev)),#
        'wd2':tf.Variable(tf.random_normal([1024,n_output],stddev=stddev))
        }
biases = {
        'bc1':tf.Variable(tf.random_normal([64],stddev=stddev)),
        'bc2':tf.Variable(tf.random_normal([128],stddev=stddev)),
        'bd1':tf.Variable(tf.random_normal([1024],stddev=stddev)),
        'bd2':tf.Variable(tf.random_normal([n_output],stddev=stddev))
        }
#%% 卷积网络的实现
def conv_basic(_input,_w,_b,_keepratio):
    # input 将输入转换为28*28的图片
    _input_r = tf.reshape(_input,shape=[-1,28,28,1])
    ## 1
    # conv layer 1
    _conv1_net = tf.nn.conv2d(_input_r,_w['wc1'],
                              strides=[1,1,1,1],padding='SAME',
                              use_cudnn_on_gpu=True)
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1_net,_b['bc1']))
    # pool layer 1
    _pool1 = tf.nn.max_pool(_conv1,ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
    _pool1_dr = tf.nn.dropout(_pool1,_keepratio)
    
    ## 2
    # conv layer 2
    _conv2_net = tf.nn.conv2d(_pool1_dr,_w['wc2'],
                              strides=[1,1,1,1],padding='SAME',
                              use_cudnn_on_gpu=True)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2_net,_b['bc2']))
    # pool layer 2
    _pool2 = tf.nn.max_pool(_conv2,ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
    _pool2_dr = tf.nn.dropout(_pool2,_keepratio)   
    
    # 将_pool2_dr转换为一维矢量
    print(_w['wd1'].get_shape().as_list())
#    _dense1 = tf.reshape(_pool2_dr,[-1,_w['wd1'].get_shape().as_list()[0]])
    _dense1 = tf.reshape(_pool2_dr,[-1,7*7*128])
    
    ## 3
    # 全连接层1
    _fc1_net = tf.add(tf.matmul(_dense1,_w['wd1']),_b['bd1'])
    _fc1 = tf.nn.relu(_fc1_net)
    _fc1_dr = tf.nn.dropout(_fc1,_keepratio)
    # 全连接层2
    _out = tf.add(tf.matmul(_fc1_dr,_w['wd2']),_b['bd2'])
    return _out

#%%
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_output])
keepratio = tf.placeholder(tf.float32)

# 预测结果
_pred = conv_basic(x,weights,biases,keepratio)
# 代价函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred,labels=y))
# 优化器
optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# 准确率
_corr = tf.equal(tf.argmax(_pred,1),tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(_corr,tf.float32))

load_from_model = False
# 保存模型
saver = tf.train.Saver(max_to_keep=3)
path = 'save/model.ckpt'
if load_from_model == False:
    #%% 启动回话,取数据进行训练和预测
    train_epoches = 5
    batch_size = 50
    display_step = 1
    batch_num = mnist.train.num_examples // batch_size
    
    begintime = datetime.datetime.now()    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(train_epoches):
            avg_cost = 0
            total_batch = mnist.train.num_examples // batch_size        
            for batch in range(total_batch):
                batch_x,batch_y = mnist.train.next_batch(batch_size)
                sess.run(optm,feed_dict={x:batch_x,y:batch_y,keepratio:0.7})
                avg_cost += sess.run(cost,
                                     feed_dict={x:batch_x,y:batch_y,keepratio:1.0}) / total_batch
            if epoch % display_step == 0:
                print('Epoch:%03d of %03d cost : %.9f' % (epoch,train_epoches,avg_cost))
                train_acc = sess.run(accr, feed_dict={x:batch_x,y:batch_y,keepratio:1.0}) 
                print('Training Accuracy : %0.3f' % (train_acc),end='\t')
                test_acc = sess.run(accr,feed_dict={x:mnist.test.images,y:mnist.test.labels,keepratio:1.0})
                print('Testing Accuracy : %0.3f' % (test_acc))
                pass
#            save_path = saver.save(sess,path+str('-')+str(epoch))      
            
            pass
        save_path = saver.save(sess,path)
        print('模型保存的位置',save_path)
    endtime = datetime.datetime.now()    
    print('耗时：',(endtime - begintime).seconds,'秒')
else:
    with tf.Session() as sess:
        saver.restore(sess,path)
        test_acc = sess.run(accr,feed_dict={x:mnist.test.images,y:mnist.test.labels,keepratio:1.0})
        print('Testing Accuracy : %0.3f' % (test_acc))
    
