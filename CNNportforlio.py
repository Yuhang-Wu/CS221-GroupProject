from __future__ import print_function
from utils import dataUtil as du
import tensorflow as tf
import numpy as np


N = 5 # depend on the N previous periods
B = 10 # number of examples per batch
dateSelected, stockPrice  = du.getData()
logreturn = du.logReturn(stockPrice)
# return for N previous periods, input, shape: [-1,K,N]
logReturn_x = du.logReturnMatrix(logreturn, N)
# return for current period
logReturn_x0 = logreturn[N:]

K = len(stockPrice[0]) # K stocks
L = len(logReturn_x0) # total number of time steps
logReturn_x_data = tf.reshape(logReturn_x, [-1,K,N,1])

x = tf.placeholder(tf.float32, shape=[None, K, N])
x_data = tf.reshape(x, [-1,K,N,1])
y_ = tf.placeholder(tf.float32, shape=[None, K])

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def avg_pool_4x1(x):
    return tf.nn.avg_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME')


# first convolution layer
W_conv1 = weight_variable([2, 2, 1, 10])
b_conv1 = bias_variable([10])

h_conv1 = tf.nn.relu(conv2d(x_data, W_conv1) + b_conv1)
h_pool1 = avg_pool_4x1(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([4, 1, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_4x1(h_conv2)

# densely connected layer
lengthW_fc1 = int(h_pool2.shape[1] * h_pool2.shape[2] * h_pool2.shape[3])
W_fc1 = weight_variable([lengthW_fc1, 2 ** K])
b_fc1 = bias_variable([2 ** K])
h_pool2_flat = tf.reshape(h_pool2, [-1, lengthW_fc1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# readout layer
W_fc2 = weight_variable([2 ** K, K])
b_fc2 = bias_variable([K])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

# train and evaluate model
action = tf.nn.softmax(logits=y_conv)
reward0 = np.multiply(action, y_)

# ========
# should there be an extra exp?***
reward0 = np.multiply(action, tf.exp(y_))
reward = tf.reduce_sum(reward0, 1)
reward_minus = -tf.reduce_mean(reward)

train_step = tf.train.AdamOptimizer(1e-4).minimize(reward_minus)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(L/B):
        historicalData = logReturn_x[i*B:(i+1)*B]
        compareReturn = logReturn_x0[i*B:(i+1)*B]
        feed_dict = {x: historicalData, y_: compareReturn}
        print('action is', sess.run(action, feed_dict = feed_dict))
        print()
        print(sess.run(-reward_minus, feed_dict = feed_dict))
        train_step.run(feed_dict = feed_dict)