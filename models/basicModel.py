from __future__ import print_function
import tensorflow as tf
from model import Model, raiseNotDefined
import os
import modelUtil as mu


class Config:
	lr = 0.0005
	dropout = 1.0

class BasicModel(Model):

	# define the placeholders (add it to self.placeholders)
	def add_placeholders(self):
		xShape = (self.D, self.N) if self.L == 1 else (self.D, self.N, self.L)
		self.placeholders = {
			'X': tf.placeholder(dtype = tf.float32, shape = xShape),
			'prevReturn': tf.placeholder(dtype = tf.float32, shape = (self.D, 1)),
			'nextReturn': tf.placeholder(dtype = tf.float32, shape = (self.D, 1)),
			'prevA': tf.placeholder(dtype = tf.float32, shape = (self.D+1, 1)),
			'mRatio': tf.placeholder(dtype = tf.float32, shape = ()),
		}
		
	# create feed dict (return it)
	def create_feed_dict(self, inputs):
		feed_dict = {
			self.placeholders[key]: inputs[key] for key in inputs
		}
		return feed_dict

	# add an action (add to self)
	def add_action(self):
		# each model must implement this method
		raiseNotDefined()
		return
		

	# create loss from action (return it)
	def add_loss(self, action):
		# calculate profit from action	
		prevReturn = self.placeholders['prevReturn']
		nextReturn = self.placeholders['nextReturn']
		mRatio = self.placeholders['mRatio']
		prevA = self.placeholders['prevA']

		transCostParams = self.transCostParams

		# calculate return (hence loss)
		profit = mu.calcProfit(action, nextReturn)
		transCost = mu.calcTransCost(action, prevA, prevReturn, transCostParams, mRatio)
		R = profit - transCost
		loss = R * (-1.0)
		return loss

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
		action = sess.run(self.action, feed_dict = feed_dict)
		return action

	# object constructor
	# D : the dimension of the portfolio,
	# N : the number of days looking back
	def __init__(self, D, N, transCostParams, L = 1):
		self.D = D
		self.N = N
		self.L = L

		self.transCostParams = {
			key: tf.constant(transCostParams[key], dtype = tf.float32) for key in transCostParams
		}

		self.build()
