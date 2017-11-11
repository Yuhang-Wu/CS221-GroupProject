# more to be clarified
# log-return as input
# softmax probability as output
# reward function: return for the future period
# Question! how to integrate the previous portfolio?

import tensorflow as tf
import numpy as np
import getStockData

N = 20 # depend on the N previous periods

dateSelected, stockPrice  = getStockData.getData()
logreturn = getStockData.logReturn(stockPrice)
# return for N previous periods, input, shape: [-1,K,N]
logReturn_x = getStockData.logReturnMatrix(logreturn, N)
# return for current period
logReturn_x0 = logreturn[:len(logReturn_x)]

K = len(stockPrice[0]) # K stocks
logReturn_data = tf.reshape(logReturn, [-1,K,N,1])


# placeholders
x = tf.placeholder(tf.float32, shape=[None, T, N])
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

def avg_pool_1*4(x):
    return tf.nn.avg_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')

# first convolution layer
W_conv1 = weight_variable([4, 1, 1, 10])
b_conv1 = bias_variable([10])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = avg_pool_4x1(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([4, 1, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_4x1(h_conv2)

# densely connected layer
W_fc1 = weight_variable([__, 2 ** K])
b_fc1 = bias_variable([2 ** K])
h_pool2_flat = tf.reshape(h_pool2, [-1, N/4/4*K*20])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# readout layer
W_fc2 = weight_variable([2 ** K, K])
b_fc2 = bias_variable([K])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# train and evaluate the model
# need modified here!!!!!!!
reward = np.multiply(tf.nn.softmax(logits=y_conv), y_)
reward = reward.sum(axis=1)
reward_minus = -reward.sum(axis=0)/N

train_step = tf.train.AdamOptimizer(1e-4).minimize(reward_minus)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        historicalData = logReturn_x[i*50:(i+1)*50]
        compareReturn = logReturn_x0[i*50:(i+1)*50]
        if i % 100 == 0:
            print(sess.run(-reward_minus, {x: historicalData, y_: compareReturn}))
        train_step.run(feed_dict={x: historicalData, y_: compareReturn})


