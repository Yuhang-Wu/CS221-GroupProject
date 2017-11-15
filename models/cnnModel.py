from __future__ import print_function
import tensorflow as tf
from model import Model, raiseNotDefined
from basicModel import BasicModel
import os
import modelUtil as mu


class Config:
	lr = 1e-4
	dropout = 0.5
	modelType = 'CNNModel'

class CnnModel(BasicModel):

	# add an action (add to self and return it)
	def add_action(self):

		X = self.placeholders['X']
		prevReturn = self.placeholders['prevReturn']
		prevA = self.placeholders['prevA']
		# define your variables here
		# calculate action
		x_data = tf.reshape(X, [-1, self.D, self.N, self.L])

		# 1st conv layer
		W_conv1 = mu.tnVariable([1, 3, 1, 5])
		b_conv1 = mu.biasVariable([5])

		h_conv1 = tf.nn.relu(mu.conv2dStide1(x_data, W_conv1) + b_conv1)
		h_pool1 = mu.avg1x1(h_conv1)

		# 2nd conv layer
		W_conv2 = mu.tnVariable([1, 3, 5, 10])
		b_conv2 = mu.biasVariable([10])

		h_conv2 = tf.nn.relu(mu.conv2dStide1(h_pool1, W_conv2) + b_conv2)
		h_pool2 = mu.avg1x1(h_conv2)

		# 1st fc layer
		wsize1 = self.N * self.D * int(W_conv2.shape[3])
		W_fc1 = mu.tnVariable([wsize1, wsize1 / 2] )
		b_fc1 = mu.biasVariable([wsize1 / 2])
		h_pool2_flat = tf.reshape(h_pool2, [-1, wsize1])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# 2nd fc layer
		W_fc2 = mu.tnVariable([wsize1 / 2, self.D + 1])
		b_fc2 = mu.biasVariable([self.D + 1])

		y_conv = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
		
		action = tf.nn.softmax(y_conv)

		action = tf.transpose(action)

		self.action = action

	# object constructor
	# D : the dimension of the portfolio,
	# N : the number of days looking back
	def __init__(self, D, N, transCostParams, L = 1):
		self.D = D
		self.N = N
		self.L = L
		self.config = Config
		self.transCostParams = {
			key: tf.constant(transCostParams[key], dtype = tf.float32) for key in transCostParams
		}

		self.build()
