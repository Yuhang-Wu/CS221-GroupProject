from model import Model, raiseNotDefined

class SlidingWindowModel(Model):

	# object constructor
	# N : the window size (how many days do we look behind)
	# D : the dimension of the portfolio 
	def __init__(self, D, N):
		self.N = N
		self.D = D

