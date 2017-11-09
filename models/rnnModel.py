from model import Model, raiseNotDefined

class RnnModel(Model):

	# object constructor
	# N : the window size (how many days do we look behind)
	# D : the dimension of the portfolio 
	def __init__(self, D):
		self.D = D

