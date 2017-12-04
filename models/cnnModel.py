from __future__ import print_function
import tensorflow as tf
import numpy as np
import os,json
import modelUtil as mu
from model import Model, raiseNotDefined
from basicModel import BasicModel



class Config:
    lr = 1e-4
    modelType = 'CNNModel'
    ConvolutionalLayers = 2
    DenseLayers = 2
    
    

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


class CnnModel(BasicModel):
    # add an action (add to self and return it)
    def add_action(self):
        # placeholder
        X = self.placeholders['X']
        x_data = tf.reshape(X, [-1, self.D, self.N, self.L]) # start from one channel
        prevA = self.placeholders['prevA'] # portfolio for last time step
        prevReturn = self.placeholders['prevReturn'] # return for last time step
    
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
        prePort = tf.reshape(prevA,[-1,1])
        W_fc12 = weight_variable([1, wsize2])
        b_fc1 = bias_variable([wsize2])
        h_pool2_flat = tf.reshape(h_pool2, [-1, wsize1])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + tf.matmul(prePort[:-1,:], W_fc12) + b_fc1)
        h_fc1_score = tf.reshape(h_fc1, [-1, self.D])

        # densely connected layer 2
        # readout layer
        W_fc2 = weight_variable([self.D, self.D])
        b_fc2 = bias_variable([self.D])
        y_conv = tf.matmul(h_fc1_score, W_fc2) + b_fc2
    
        # produce portfolio
        currentPortfolio = tf.nn.softmax(logits=y_conv)
    
        action = tf.concat([currentPortfolio, [[1]]],1)
        self.action = tf.reshape(action,[self.D+1,1])

        
    def get_model_info(self):
        model_info = {
            'lr': self.config.lr,
            'model_type': self.config.modelType,
            'ConvolutionalLayers': self.config.ConvolutionalLayers,
            'DenseLayers': self.config.DenseLayers
        }
        return json.dumps(model_info)
    
    # object constructor
    # D : the dimension of the portfolio,
    # N : the number of days looking back
    # L : the number of data points per time step
    def __init__(self, D, N, transCostParams, L = 1):
        self.D = D
        self.N = N
        self.L = L
        self.config = Config
        self.transCostParams = {
            key: tf.constant(transCostParams[key], dtype = tf.float32) for key in transCostParams
        }

        self.build()
