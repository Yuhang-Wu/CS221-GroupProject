import tensorflow as tf
import numpy as np
from utils import dataUtil

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


def CNN(stockPrice, Time, c):
    logreturn = dataUtil.logReturn(stockPrice)
    N = 10 # depend on the N previous periods
    # return for N previous periods, input, shape: [-1,L,N]
    logReturn_x = dataUtil.logReturnMatrix(logreturn, N)
    # return for current period
    logReturn_x0 = logreturn[N:]
    
    L = len(stockPrice[0]) # L stocks
    B = 1 # batch size
    
    # training and testing data
    Time = np.array(Time) - 1 - N
    TestIndex = Time
    TrainIndex = range(Time[0])
    train_x = [logReturn_x[i] for i in TrainIndex]
    train_y = [logReturn_x0[i] for i in TrainIndex]
    test_x = [logReturn_x[i] for i in TestIndex]
    test_y = [logReturn_x0[i] for i in TestIndex]
    
    # get CNN model
    
    # placeholder
    x = tf.placeholder(tf.float32, shape=[None, L, N])
    x_data = tf.reshape(x, [-1,L,N,1]) # start from one channel
    y_ = tf.placeholder(tf.float32, shape=[None, L])
    previousPortfolio = tf.placeholder(tf.float32, shape=[None, L]) # portfolio for last time step
    previousReturn = tf.placeholder(tf.float32, shape=[None]) # return for last time step
    
    # first convolution layer
    W_conv1 = weight_variable([1, 3, 1, 10])
    b_conv1 = bias_variable([10])
    h_conv1 = tf.nn.relu(conv2d(x_data, W_conv1) + b_conv1)
    h_pool1 = avg_pool_4x1(h_conv1)
    
    # second convolutional layer
    W_conv2 = weight_variable([1, 3, 10, 5])
    b_conv2 = bias_variable([5])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = avg_pool_4x1(h_conv2)
    
    # densely connected layer 1
    wsize1 = int(h_pool2.shape[2] * h_pool2.shape[3])
    wsize2 = 1
    W_fc1 = weight_variable([wsize1, wsize2] )
    prePort = tf.reshape(previousPortfolio,[-1,1])
    W_fc12 = weight_variable([1, wsize2])
    b_fc1 = bias_variable([wsize2])
    h_pool2_flat = tf.reshape(h_pool2, [-1, wsize1])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + tf.matmul(prePort, W_fc12) + b_fc1)
    h_fc1_score = tf.reshape(h_fc1, [-1, L])
    
    # densely connected layer 2
    # readout layer
    W_fc2 = weight_variable([L, L])
    b_fc2 = bias_variable([L])
    y_conv = tf.matmul(h_fc1_score, W_fc2) + b_fc2
    
    # produce portfolio
    currentPortfolio = tf.nn.softmax(logits=y_conv)
    
    # define loss function
    reward0 = np.multiply(currentPortfolio, y_)
    reward = tf.reduce_sum(reward0, 1)
    # calculate return including transaction cost
    flag = 0
    for j in xrange(L):
        tmp = currentPortfolio[:,j]-previousPortfolio[:,j]*tf.divide(tf.exp(x[:,j,-1]),(1.0+previousReturn))
        reward = reward - c[j+1] * np.abs(tmp)
        if tmp != 0:
            flag = 1
    if flag:
        reward = reward - c[0]
    
    # loss function: reward_minus
    reward_minus = -tf.reduce_prod(reward+1)
    


    train_step = tf.train.AdamOptimizer(1e-4).minimize(reward_minus)
    test_step = tf.train.AdamOptimizer(1e-4).minimize(reward_minus)
    
    # train and test model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # train model
        pre_train_accumulate = 1
        for k in range(500):
            PrePortfolio = np.zeros([B,L])
            PreReturn = np.zeros(B)
            train_accumulate = 1
            for i in TrainIndex:
                historicalData = logReturn_x[i*B:(i+1)*B]
                compareReturn = logReturn_x0[i*B:(i+1)*B]
                currentReturn = sess.run(reward, {x: historicalData, y_: compareReturn, previousPortfolio: PrePortfolio, previousReturn: PreReturn} )
                train_accumulate = train_accumulate * (sess.run(-reward_minus,{x: historicalData,y_: compareReturn, previousPortfolio: PrePortfolio, previousReturn: PreReturn}))
                train_step.run(feed_dict={x: historicalData, y_: compareReturn,previousPortfolio: PrePortfolio, previousReturn: PreReturn})
                prePortfolio = sess.run(currentPortfolio, {x: historicalData,y_: compareReturn, previousPortfolio: PrePortfolio, previousReturn: PreReturn})
                preReturn = currentReturn
            if np.abs(train_accumulate - pre_train_accumulate)<1e-8:
                break
            else:
                pre_train_accumulate = train_accumulate
            print 'epoch{0}, the training accumulated return is {1}.'.format(k, train_accumulate)
    
        # test model
        test_accumulate = 1
        PrePortfolio = np.zeros([B,L])
        PreReturn = np.zeros(B)
        testReturn = []
        for i in TestIndex:
            historicalData = logReturn_x[i*B:(i+1)*B]
            compareReturn = logReturn_x0[i*B:(i+1)*B]
            currentReturn = sess.run(reward, {x: historicalData, y_: compareReturn, previousPortfolio: PrePortfolio, previousReturn: PreReturn} )
            testReturn.append(currentReturn)
            test_accumulate = test_accumulate * (sess.run(-reward_minus, {x: historicalData,y_: compareReturn, previousPortfolio: PrePortfolio, previousReturn: PreReturn}))
            test_step.run(feed_dict={x: historicalData, y_: compareReturn, previousPortfolio: PrePortfolio, previousReturn: PreReturn})
            prePortfolio = sess.run(currentPortfolio,  {x: historicalData,y_: compareReturn,  previousPortfolio: PrePortfolio, previousReturn: PreReturn})
            preReturn = currentReturn

return testReturn


"""
# get data (date, stockPrice)
dateSelected, stockPrice = dataUtil.getData()

# get time for baseline estimation
Time = range(10+(len(dateSelected)-10)/2+1,len(dateSelected))

# Date for estimated return period (startDate,endDate) =  (dateSelected[Time[i]-1],dateSelected[Time[i]])
Date = [(dateSelected[i-1][0],dateSelected[i][0]) for i in Time]

# parameters for transaction cost
c = np.zeros(len(stockPrice[-1])+1) + 0.0001

# estimated period return for corresponding date
estimateReturn = CNN(stockPrice, Time, c)
"""






