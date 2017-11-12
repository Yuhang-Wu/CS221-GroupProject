from models import dummyModel as dm
import numpy as np
import tensorflow as tf
import sys, os, csv
from utils import readin, yfReader, dataUtil as du

DATA_PATH = 'data/sp10/'

def main():
	dummyModelTrainingTrial(DATA_PATH)
	

def dummyModelTrainingTrial(datapath1):
	stockPrices = getPricesFromPath(datapath1)

	D = stockPrices.shape[0]
	N = 3
	c = 0.0001
	epochs = 50

	transCostParams = {
		'c': np.array([ [c] for _ in range(D) ]),
		'c0': c
	}

	curModel = dm.DummyModel(D, N, transCostParams)
	init_op = tf.global_variables_initializer()

	totalIters = stockPrices.shape[1] - N

	prevLoss = 0.0
	prevA = np.array( map(lambda x : [x], [0.0 for _ in range(D)] + [1.0]) )

	
	with tf.Session() as sess:

		sess.run(init_op)
		for i in range(epochs):
			allActions = []
			allLosses = []
			for t in range(totalIters):

				prevS = stockPrices[:, t : t + N]
				nextS = stockPrices[:, t + N : t + N + 1]
				mRatio = du.loss2mRatio(prevLoss)
				inputs = {
					'prevA': prevA,
					'prevS': prevS,
					'nextS': nextS,
					'mRatio': mRatio
				}
				
				curA, curLoss = curModel.train(inputs, sess)

				allActions.append(curA)
				allLosses.append(curLoss)

				prevLoss = curLoss
				prevA = curA
			totalLoss = sum(allLosses)
			print(i, 'th epoch')
			print('total earnings:')
			print(-1.0*totalLoss)

	#print(allActions)
	#print(allLosses)

	


def getPricesFromPath(datapath1): 
	allfilecontents = readin.readCsvFromPath(datapath1)
	dateSelected = du.selectDate(allfilecontents)
	stockPricesList = du.getStockPrice(allfilecontents, dateSelected)
	stockPrices = np.array(stockPricesList).T
	return stockPrices

if __name__=='__main__':
	main()



