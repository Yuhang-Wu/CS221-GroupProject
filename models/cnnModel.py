from __future__ import print_function
import tensorflow as tf
from model import Model, raiseNotDefined
from basicModel import BasicModel
import os
import modelUtil as mu


class Config:
	lr = 0.002
	dropout = 0.5

class CnnModel(BasicModel):

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
		

		X = self.placeholders['X']
		prevReturn = self.placeholders['prevReturn']
		prevA = self.placeholders['prevA']

		# calculate action
		# helpful functions:
		# tf.matmul, tf.nn.softmax

		U = tf.matmul(self.W1, X)
		U = tf.nn.relu(U)

		alpha = tf.reduce_mean(U, axis = 1, keep_dims = True)

		beta = tf.matmul(self.W2, prevA)
		beta = tf.nn.relu(beta)
		
		alphabeta = tf.concat([alpha, beta], axis = 0)

		action = tf.matmul(self.W3, alphabeta)
		#action = tf.sigmoid(action)
		action = tf.nn.softmax(action, dim = 0)

		self.action = action
