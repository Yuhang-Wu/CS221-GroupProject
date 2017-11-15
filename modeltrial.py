from __future__ import print_function
import numpy as np
import tensorflow as tf
from models import dummyModel as dm
from utils import readin, yfReader, dataUtil as du


DATA_PATH = 'data/sp10/'

def main():
	dummyModelTrainingTrial()


def dummyModelTrainingTrial():
	stockPrices = getPricesFromPath(DATA_PATH)
	N = 5
	c = 0.0001
	epochs = 200

	D = stockPrices.shape[1]
	
	returnMatrix = du.logReturn(stockPrices)
	prevReturnMatrix = du.extendDimension(returnMatrix[N-2:-1])
	nextReturnMatrix = du.extendDimension(returnMatrix[N-1:])
	returnTensor = du.preprocess(stockPrices, N)
	
	print('num of datapoints', returnTensor.shape[0])
	transCostParams = {
		'c': np.array([ [c] for _ in range(D) ]),
		'c0': c
	}
	
	curModel = dm.DummyModel(D, N, transCostParams)

	totalIters = returnTensor.shape[0]

	prevLoss = 0.0
	prevA = du.getInitialAllocation(D)
	
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		for i in range(epochs):
			allActions = []
			allLosses = []
			for t in range(totalIters):
				
				mRatio = du.loss2mRatio(prevLoss)

				inputs = {
					'X': returnTensor[t],
					'prevReturn': prevReturnMatrix[t],
					'nextReturn': nextReturnMatrix[t],
					'prevA': prevA,
					'mRatio': mRatio
				}
				
				curA, curLoss = curModel.train(inputs, sess)

				allActions.append(curA)
				allLosses.append(curLoss)

				prevLoss = curLoss
				prevA = curA
			#print(allActions[-5])
			totalLoss = sum(allLosses)
			growthRates = map(lambda x: 1-x, allLosses)
			totalGR = du.prod(growthRates)
			if i%10 == 0:
				print(i, 'th epoch')
				print('total growth rate:')
				print(totalGR)
				print()

	#print(allActions)
	#print(allLosses)

	
def getPricesFromPath(datapath1): 
	allfilecontents = readin.readCsvFromPath(datapath1)
	dateSelected = du.selectDate(allfilecontents, 'week')
	stockPricesList = du.getStockPrice(allfilecontents, dateSelected)
	stockPrices = np.array(stockPricesList)
	return stockPrices

if __name__=='__main__':
	main()



