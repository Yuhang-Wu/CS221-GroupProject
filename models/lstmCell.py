from __future__ import print_function
from __future__ import absolute_import
import sys
import tensorflow as tf
import numpy as np

def sigma_h(x):
    return x

class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, hiddenstate, scope=None):
        scope = scope or type(self).__name__
        
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(scope):
            W_f = tf.get_variable('W_f',
                                  [self.state_size, self.state_size],
                                  initializer = initializer)
            W_i = tf.get_variable('W_i',
                                  [self.state_size, self.state_size],
                                  initializer = initializer)
            W_o = tf.get_variable('W_o',
                                  [self.state_size, self.state_size],
                                  initializer = initializer)
            W_c = tf.get_variable('W_c',
                                  [self.state_size, self.state_size],
                                  initializer = initializer)

            U_f = tf.get_variable('U_f',
                                  [self.input_size, self.state_size],
                                  initializer = initializer)
            U_i = tf.get_variable('U_i',
                                  [self.input_size, self.state_size],
                                  initializer = initializer)
            U_o = tf.get_variable('U_o',
                                  [self.input_size, self.state_size],
                                  initializer = initializer)
            U_c = tf.get_variable('U_c',
                                  [self.input_size, self.state_size],
                                  initializer = initializer)

            b_f = tf.get_variable('b_f',
                                  [self.state_size, ],
                                  initializer = initializer)
            b_i = tf.get_variable('b_i',
                                  [self.state_size, ],
                                  initializer = initializer)
            b_o = tf.get_variable('b_o',
                                  [self.state_size, ],
                                  initializer = initializer)
            b_c = tf.get_variable('b_c',
                                  [self.state_size, ],
                                  initializer = initializer)

            f_t = tf.nn.sigmoid(tf.matmul(inputs, U_f) + tf.matmul(state, W_f) + b_f)
            i_t = tf.nn.sigmoid(tf.matmul(inputs, U_i) + tf.matmul(state, W_i) + b_i)
            o_t = tf.nn.sigmoid(tf.matmul(inputs, U_o) + tf.matmul(state, W_o) + b_o)
            c_t = tf.nn.tanh(tf.matmul(inputs, U_c) + tf.matmul(state, W_c) + b_c)
            c_t = tf.multiply(f_t, hiddenstate) + tf.multiply(i_t, c_t)
            h_t = tf.multiply(o_t, sigma_h(c_t))

        new_state = h_t
        new_hiddenstate = c_t
        return new_state, new_hiddenstate


'''
        with tf.variable_scope(scope):
            W_r = tf.get_variable('W_r',
                                  [self.state_size, self.state_size],
                                  initializer = initializer)
            U_r = tf.get_variable('U_r',
                                  [self.input_size, self.state_size],
                                  initializer = initializer)
            b_r = tf.get_variable('b_r',
                                  [self.state_size, ],
                                  initializer = initializer)
            W_z = tf.get_variable('W_z',
                                  [self.state_size, self.state_size],
                                  initializer = initializer)
            U_z = tf.get_variable('U_z',
                                  [self.input_size, self.state_size],
                                  initializer = initializer)
            b_z = tf.get_variable('b_z',
                                  [self.state_size, ],
                                  initializer = initializer)
            W_o = tf.get_variable('W_o',
                                  [self.state_size, self.state_size],
                                  initializer = initializer)
            U_o = tf.get_variable('U_o',
                                  [self.input_size, self.state_size],
                                  initializer = initializer)
            b_o = tf.get_variable('b_o',
                                  [self.state_size, ],
                                  initializer = initializer)

            z = tf.nn.sigmoid(tf.matmul(inputs, U_z) + tf.matmul(state, W_z) + b_z)
            r = tf.nn.sigmoid(tf.matmul(inputs, U_r) + tf.matmul(state, W_r) + b_r)
            h_hat = tf.nn.tanh(tf.matmul(inputs, U_o) + tf.matmul(r * state, W_o) + b_o)
            new_state = z * state + (1 - z) * h_hat

        new_hiddenstate = hiddenstate
        return new_state, new_hiddenstate
'''
