from model import Model, raiseNotDefined

class RnnModel(Model):

	# object constructor
	# N : the window size (how many days do we look behind)
	# D : the dimension of the portfolio 
	def __init__(self, D):
		self.D = D

	# create feed dict (add it to self)
	def create_feed_dict(self, inputs):
		raiseNotDefined()

	# define the variables (add it to self)
	def add_placeholders(self):
		raiseNotDefined()

	# add an action (return it)
	def add_action(self):
		raiseNotDefined()

	# create loss from action (return it)
	def add_loss(self, action):
		raiseNotDefined()

	# define how to train from loss (return it)
	def add_train_op(self, loss):
		raiseNotDefined()

	# train the model with 1 iteration
	def train(self, sess):
		raiseNotDefined()

	# get the action of the next time step
	def get_action(self, sess):
		raiseNotDefined()