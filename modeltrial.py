from __future__ import print_function
import sys, os, csv
from models import dummyModel as dm
import numpy as np
import tensorflow as tf
from utils import readin, yfReader, dataUtil as du


DATA_PATH = 'data/sp10/'

def main():
	dummyModelTrainingTrial(DATA_PATH)
	

def dummyModelTrainingTrial(datapath1):
	stockPrices = getPricesFromPath(datapath1)

	D = stockPrices.shape[0]
	N = 3
	c = 0.0001
	epochs = 10
	print('num of datapoints', stockPrices.shape[1])
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
			growthRates = map(lambda x: 1-x, allLosses)
			totalGR = 1.0
			for gr in growthRates:
				totalGR *= gr
			print(i, 'th epoch')
			#print('total linear earnings:')
			#print(-1.0*totalLoss)
			print('total growth rate:')
			print(totalGR)
			print()

	#print(allActions)
	#print(allLosses)

	
def getPricesFromPath(datapath1): 
	allfilecontents = readin.readCsvFromPath(datapath1)
	dateSelected = du.selectDate(allfilecontents, 'week')
	stockPricesList = du.getStockPrice(allfilecontents, dateSelected)
	stockPrices = np.array(stockPricesList).T
	return stockPrices

if __name__=='__main__':
	main()



