from __future__ import print_function
import tensorflow as tf
from model import Model, raiseNotDefined
from basicModel import BasicModel
import os, json
import modelUtil as mu
from rnnCell import RNNCell 
from gruCell import GRUCell
from lstmCell import LSTMCell

class Config:
	lr = 1e-3
	dropout = 0.5
	modelType = 'RNNModel'
	cellType = 'rnn'
	hiddenSize = 15
	transformSize = 10

class RnnModel(BasicModel):
	# add an action (add to self and return it)
	def add_action(self):
		# define your variables here

		X = self.placeholders['X']
		prevReturn = self.placeholders['prevReturn']
		prevA = self.placeholders['prevA']
		print(X.shape)
		cellType = self.config.cellType
		if cellType == 'rnn':
			cell = RNNCell(self.L, self.config.hiddenSize)
		elif cellType == 'gru':
			cell = GRUCell(self.L, self.config.hiddenSize)
		elif cellType == 'lstm':
			cell = LSTMCell(self.L, self.config.hiddenSize)
		else:
			assert False, "Cell type undefined"
		h = tf.zeros([self.D, self.config.hiddenSize], dtype = tf.float32)
		hh = tf.zeros([self.D, self.config.hiddenSize], dtype = tf.float32)
		states = []
		hiddenstates = []
			

		with tf.variable_scope("RNN"):
			for t in range(self.N):
				if t>=1:
					tf.get_variable_scope().reuse_variables()
				h, hh = cell(X[:, t, :], h, hh)
				states.append(h)
				hiddenstates.append(hh)
		# calculate action based on all hidden states

		initializer = tf.contrib.layers.xavier_initializer()
		W_fc1 = tf.get_variable('W_fc1',
		                      [self.config.hiddenSize, self.D],
		                      initializer = initializer)
		b_fc1 = tf.get_variable('b_fc1',
		                      [self.D, ],
		                      initializer = initializer)

		W_fc2 = tf.get_variable('W_fc2',
		                      [self.D, 1],
		                      initializer = initializer)
		b_fc2 = tf.get_variable('b_fc2',
		                      [1, ],
		                      initializer = initializer)

		y_fc1 = tf.nn.sigmoid(tf.matmul(states[-1], W_fc1) + b_fc1)
		y_fc2 = tf.nn.relu(tf.matmul(y_fc1, W_fc2) + b_fc2)

		action = mu.addBias(y_fc2)
		action = tf.nn.softmax(action, dim = 0)
		self.action = action

	def get_model_info(self):
		model_info = {
			'lr': self.config.lr,
			'dropout': self.config.dropout,
			'model_type': self.config.modelType,
			'cell_type': self.config.cellType,
			'hidden_size': self.config.hiddenSize
		}
		print("model info")
		print(json.dumps(model_info))
		print()

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
