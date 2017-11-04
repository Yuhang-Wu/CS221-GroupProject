
def raiseNotDefined():
	raise NotImplementedError("Each Model must re-implement this method.")

class Model(object):
	def __init__(self):
		pass

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

	# build the computation graph (add them to self)
	# called after initializing an instance of the object
	def build(self):
		self.add_placeholders()
		self.action = self.add_action()
		self.loss = self.add_loss(self.action)
		self.train_op = self.add_train_op(self.loss)

	# train the model with 1 iteration
	def train(self, sess):
		raiseNotDefined()

	# get the action of the next time step
	def get_action(self, sess):
		raiseNotDefined()