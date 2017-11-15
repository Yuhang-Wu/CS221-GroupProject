import tensorflow as tf
import numpy as np
from utils import dataUtil

N = 8 # depend on the N previous periods
B = 1 # batch size

dateSelected, stockPrice  = dataUtil.getData()

logreturn = dataUtil.logReturn(stockPrice)

# return for N previous periods, input, shape: [-1,K,N]
logReturn_x = dataUtil.logReturnMatrix(logreturn, N)
# return for current period
logReturn_x0 = logreturn[N:]

K = len(stockPrice[0]) # K stocks
logReturn_x_data = tf.reshape(logReturn_x, [-1,K,N,1])

x = tf.placeholder(tf.float32, shape=[None, K, N])
x_data = tf.reshape(x, [-1,K,N,1])
y_ = tf.placeholder(tf.float32, shape=[None, K])
previousPortfolio = tf.placeholder(tf.float32, shape=[None, K]) # portfolio for last time step
previousReturn = tf.placeholder(tf.float32, shape=[None]) # return for the last time step

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
    return tf.nn.avg_pool(x, ksize=[1, 1, 1, 1],
                          strides=[1, 1, 1, 1], padding='SAME')

# first convolution layer
W_conv1 = weight_variable([1, 3, 1, 5])
b_conv1 = bias_variable([5])

h_conv1 = tf.nn.relu(conv2d(x_data, W_conv1) + b_conv1)
h_pool1 = avg_pool_4x1(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([1, 3, 5, 10])
b_conv2 = bias_variable([10])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_4x1(h_conv2)
print h_pool2.shape

# densely connected layer 1
W_fc1 = weight_variable([150,75])
wsize1 = int(h_pool2.shape[2] * h_pool2.shape[3])
W_fc1 = weight_variable([wsize1, wsize1 / 2] )
b_fc1 = bias_variable([wsize1 / 2])
h_pool2_flat = tf.reshape(h_pool2, [-1, wsize1])
# tf.reshape(h_pool2, [-1, lengthW_fc1]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# densely connected layer 2
# readout layer
# lengthW_fc1 = int(h_pool2.shape[1] * h_pool2.shape[2] * h_pool2.shape[3])
W_fc2 = weight_variable([wsize1 / 2, 1])
b_fc2 = bias_variable([1])
y_conv = tf.reshape(tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2), [-1, int(h_pool2.shape[1])])

currentPortfolio = tf.nn.softmax(logits=y_conv)

reward0 = np.multiply(currentPortfolio, y_)
reward = tf.reduce_sum(reward0, 1)

# calculate return including transaction cost

c = np.zeros(K+1) + 0.001 # transaction cost coefficients

flag = 0
for j in xrange(K):
    tmp = currentPortfolio[:,j]-previousPortfolio[:,j]*tf.divide(tf.exp(x[:,j,0]),(1.0+previousReturn))
    reward = reward - c[j+1] * np.abs(tmp)
    if tmp != 0:
        flag = 1
if flag:
    reward = reward - c[0]



reward_minus = -tf.reduce_mean(reward)

train_step = tf.train.AdamOptimizer(1e-4).minimize(reward_minus)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    accumulate = 1;
    PrePortfolio = np.zeros([1,K])
    PreReturn = [0]
    for i in range(len(logReturn_x0)/B):
        historicalData = logReturn_x[i*B:(i+1)*B]
        compareReturn = logReturn_x0[i*B:(i+1)*B]
        currentReturn = sess.run(reward, {x: historicalData, y_: compareReturn, \
                                 previousPortfolio: PrePortfolio, previousReturn: PreReturn} )
        print currentReturn
        accumulate = accumulate * (1 + sess.run(-reward_minus,\
                                                 {x: historicalData,y_: compareReturn, \
                                                 previousPortfolio: PrePortfolio, previousReturn: PreReturn}))
        train_step.run(feed_dict={x: historicalData, y_: compareReturn,\
                        previousPortfolio: PrePortfolio, previousReturn: PreReturn})
        prePortfolio = sess.run(currentPortfolio, \
                                 {x: historicalData,y_: compareReturn, \
                                 previousPortfolio: PrePortfolio, previousReturn: PreReturn})
        preReturn = currentReturn

print accumulate
