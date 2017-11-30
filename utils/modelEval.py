import math
import numpy as np



# calculate total return and sharp ration
def totalReturn(Re):
	result = 1
	for re in Re:
		result *= (1+re)
	return result

def sharp_ratio(Re):
	Re = np.array(Re)
	return np.mean(Re)/np.std(Re)

def evaluateModel(Re_bl, Re_oc, Re_rnn, Re_cnn):
	print 'totalReturn \t sharp_ratio'
	print 'baseline', totalReturn(Re_bl), sharp_ratio(Re_bl)
	print 'oracle', totalReturn(Re_oc), sharp_ratio(Re_oc)
	print 'cnn', totalReturn(Re_cnn), sharp_ratio(Re_cnn)
	print 'rnn', totalReturn(Re_rnn), sharp_ratio(Re_rnn)


