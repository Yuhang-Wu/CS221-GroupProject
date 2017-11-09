from __future__ import print_function
import tensorflow as tf
from model import Model, raiseNotDefined
import os

cwd = os.getcwd()

print(cwd)

class Config:
	lr = 0.001
	dropout = 0.5

class DummyModel(Model):

	# define the placeholders (add it to self.placeholders)
	def add_placeholders(self):
		self.placeholders = {
			'prevP' 	: tf.placeholder(dtype = tf.float32, shape = (self.D+1, 1)),
			'prevPrices': tf.placeholder(dtype = tf.float32, shape = (self.D, self.N)),
			'nextPrices': tf.placeholder(dtype = tf.float32, shape = (self.D, 1))
		}
		'''
		self.prevP = tf.placeholder(dtype = tf.float32, shape = (self.D+1,))
		self.prevPrices = tf.placeholder(dtype = tf.float32, shape = (self.D, self.N))
		self.nextPrices = tf.placeholder(dtype = tf.float32, shape = (self.D, ))
		'''

	# create feed dict (return it)
	''' specs
	inputs:
		a dictionary of all the inputs
		keys: 	'prevP': a np array of shape (D+1, 1)
			   	'prevPrices': a np array of shape (D, N)
				'nextPrices': a np array of shape (D, 1)
	'''
	def create_feed_dict(self, inputs):
		feed_dict = {
			self.placeholder[key]: inputs[key] for key in inputs
		}
		return feed_dict

	# add an action (return it)
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
		

		prevPrices = self.placeholders['prevPrices']
		nextPrices = self.placeholders['nextPrices']
		prevP = self.placeholders['prevP']

		# calculate action
		# helpful functions :
		# tf.matmul, tf.nn.softmax

		U = tf.matmul(self.W1, prevPrices)
		U = tf.nn.relu(U)

		alpha = tf.reduce_mean(U, axis = 1, keep_dims = True)

		beta = tf.matmul(self.W2, prevP)
		beta = tf.nn.relu(beta)
		
		alphabeta = tf.concat([alpha, beta], axis = 0)

		action = tf.matmul(self.W3, alphabeta)
		action = tf.nn.softmax(action, dim = 0)

		print('W1', self.W1.shape)
		print('W2', self.W2.shape)
		print('W3', self.W3.shape)
		print('prevPrices', prevPrices.shape)
		print('nextPrices', nextPrices.shape)
		print('prevP', prevP.shape)
		print('U', U.shape)
		print('alpha', alpha.shape)
		print('beta', beta.shape)
		print('alphabeta', alphabeta.shape)
		print('action', action.shape)

		return action

	# create loss from action (return it)
	def add_loss(self, action):
		# calculate profit from action
		prevP = self.placeholders['prevP']
		prevPrices = self.placeholders['prevPrices']
		nextPrices = self.placeholders['nextPrices']

		# helpful functions: 
		# tf.matmul, tf.abs, tf.reduce_sum
		profit = tf.reduce_sum(action)
		# it's ok to use tf.constant here 
		return -1 * profit

	# define how to train from loss (return it)
	def add_train_op(self, loss):
		optimizer = tf.train.GradientDescentOptimizer(Config.lr)
		train_op = optimizer.minimize(loss)
		return train_op

	# train the model with 1 iteration
	def train(self, inputs, sess):
		feed_dict = self.create_feed_dict(inputs)
		_, loss = sess.run([self.train_op, self.loss], feed_dict = feed_dict)
		return loss

	# get the action of the next time step
	def get_action(self, inputs, sess):
		feed_dict = self.create_feed_dict(inputs)
		action = self.add_action()
		result = sess.run(action, feed_dict = feed_dict)
		return result

	# object constructor
	# D : the dimension of the portfolio 
	def __init__(self, D, N):
		self.D = D
		self.N = N

		print('D', D)
		print('N', N)

		self.build()
