from __future__ import print_function
import tensorflow as tf
from model import Model, raiseNotDefined
import os
import modelUtil as mu


class Config:
	lr = 0.0001
	dropout = 0.5

class DummyModel(Model):

	# define the placeholders (add it to self.placeholders)
	def add_placeholders(self):
		self.placeholders = {
			'prevA': tf.placeholder(dtype = tf.float32, shape = (self.D+1, 1)),
			'prevS': tf.placeholder(dtype = tf.float32, shape = (self.D, self.N)),
			'nextS': tf.placeholder(dtype = tf.float32, shape = (self.D, 1)),
			'mRatio': tf.placeholder(dtype = tf.float32, shape = ())
		}

	# create feed dict (return it)
	''' specs
	inputs:
		a dictionary of all the inputs
		keys: 	'prevA': a np array of shape (D+1, 1)
			   	'prevS': a np array of shape (D, N)
				'nextS': a np array of shape (D, 1)
	'''
	def create_feed_dict(self, inputs):
		feed_dict = {
			self.placeholders[key]: inputs[key] for key in inputs
		}
		return feed_dict

	# add an action (add to self and return it)
	def add_action(self):
		# define your variables here
		self.W1 = tf.get_variable('W1',
                              [self.D, self.D],
                              initializer = tf.contrib.layers.xavier_initializer())
		self.W2 = tf.get_variable('W2',
                              [self.D, self.D + 1],
                              initializer = tf.contrib.layers.xavier_initializer())
		self.W3 = tf.get_variable('W3',
                              [self.D + 1, self.D * 2],
                              initializer = tf.contrib.layers.xavier_initializer())
		

		prevS = self.placeholders['prevS']
		nextS = self.placeholders['nextS']
		prevA = self.placeholders['prevA']

		# calculate action
		# helpful functions :
		# tf.matmul, tf.nn.softmax

		U = tf.matmul(self.W1, prevS)
		U = tf.nn.relu(U)

		alpha = tf.reduce_mean(U, axis = 1, keep_dims = True)

		beta = tf.matmul(self.W2, prevA)
		beta = tf.nn.relu(beta)
		
		alphabeta = tf.concat([alpha, beta], axis = 0)

		action = tf.matmul(self.W3, alphabeta)
		action = tf.sigmoid(action)
		action = tf.nn.softmax(action, dim = 0)

		print('W1', self.W1.shape)
		print('W2', self.W2.shape)
		print('W3', self.W3.shape)
		print('prevS', prevS.shape)
		print('nextS', nextS.shape)
		print('prevA', prevA.shape)
		print('U', U.shape)
		print('alpha', alpha.shape)
		print('beta', beta.shape)
		print('alphabeta', alphabeta.shape)
		print('action', action.shape)

		self.action = action
		return action

	# create loss from action (return it)
	def add_loss(self, action):
		# calculate profit from action
		prevA = self.placeholders['prevA']
		prevS = self.placeholders['prevS']
		nextS = self.placeholders['nextS']
		mRatio = self.placeholders['mRatio']
		transCostParams = self.transCostParams
		# helpful functions: 
		# tf.matmul, tf.abs, tf.reduce_sum
		R = mu.calcReturn(prevA, prevS, nextS, mRatio, action, transCostParams, self.D, self.N)
		#R = tf.reduce_sum(action)
		
		return -1 * R

	# define how to train from loss (return it)
	def add_train_op(self, loss):
		optimizer = tf.train.AdamOptimizer(Config.lr)
		train_op = optimizer.minimize(loss)
		return train_op

	# train the model with 1 iteration
	def train(self, inputs, sess):
		feed_dict = self.create_feed_dict(inputs)
		action = sess.run(self.action, feed_dict = feed_dict)
		loss, _ = sess.run([self.loss, self.train_op], feed_dict = feed_dict)
		return action, loss

	# get the action of the current time step
	def get_action(self, inputs, sess):
		feed_dict = self.create_feed_dict(inputs)
		action = self.add_action()
		result = sess.run(action, feed_dict = feed_dict)
		return result

	# object constructor
	# D : the dimension of the portfolio 
	def __init__(self, D, N, transCostParams):
		self.D = D
		self.N = N
		self.transCostParams = {
			key: tf.constant(transCostParams[key], dtype = tf.float32) for key in transCostParams
		}

		print('D', D)
		print('N', N)
		print('transCostParams', transCostParams)

		self.build()
