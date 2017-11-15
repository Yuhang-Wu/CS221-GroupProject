from __future__ import print_function
import numpy as np
import tensorflow as tf
from models import modelUtil as mu, dummyModel as dm
from utils import readin, yfReader, dataUtil as du


DATA_PATH = 'data/sp10/'

def main():
	dummyModelTrainingTrial()

def dummyModelTrainingTrial():
	dateSelected, stockPriceList = du.getData(DATA_PATH)
	stockPrices = np.array(stockPriceList)

	N = 5
	c = 0.0001
	epochs = 200
	D = stockPrices.shape[1]
	transCostParams = {
		'c': np.array([ [c] for _ in range(D) ]),
		'c0': c
	}

	# all the inputs!!
	returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrices, N, 'vsToday')

	# define model
	curModel = dm.DummyModel(D, N, transCostParams)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(epochs):
			allActions, growthRates = mu.train1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess)
			totalGR = du.prod(growthRates)
			if i%10 == 0:
				print(i, 'th epoch')
				print('total growth rate:')
				print(totalGR)
				print()

if __name__=='__main__':
	main()



