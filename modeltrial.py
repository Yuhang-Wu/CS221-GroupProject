from __future__ import print_function
import numpy as np
import tensorflow as tf
from models import modelUtil as mu
from models import dummyModel as dm
from models import cnnModel as cm
from models import rnnModel as rm
from utils import readin, yfReader, dataUtil as du

DATA_PATH = 'data/sp10/'

N = 5
c = 0.001
epochs = 400

def main():
	rnnModelTrainingTrial()

def cnnModelTrainingTrial():
	dateSelected, stockPriceList = du.getData(DATA_PATH, frequency = 'month')
	stockPrices = np.array(stockPriceList)

	D = stockPrices.shape[1]
	transCostParams = {
		'c': np.array([ [c] for _ in range(D) ]),
		'c0': c
	}

	# all the inputs!!
	returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrices, N, 'vsYesterday')

	#print(returnTensor[0])
	#print(returnTensor[1])
	# define model
	curModel = cm.CnnModel(D, N, transCostParams)
	curModel.get_model_info()

	#quit()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(epochs):
			allActions, growthRates = mu.train1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess)
			totalGR = du.prod(growthRates)
			if i%10 == 0:
				#print(len(allActions))
				#print(allActions[4])
				print(i, 'th epoch')
				print('total growth rate:')
				print(totalGR)
				print()

def rnnModelTrainingTrial():
	dateSelected, stockPriceList = du.getData(DATA_PATH, getAll = True)
	stockPrices = np.array(stockPriceList)

	L= 4
	D = stockPrices.shape[1]
	transCostParams = {
		'c': np.array([ [c] for _ in range(D) ]),
		'c0': c
	}
	
	# all the inputs!!
	returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrices, N, 'vsYesterday', L)

	# define model
	curModel = rm.RnnModel(D, N, transCostParams, L = 4)
	curModel.get_model_info()
	#quit()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(epochs):
			allActions, growthRates = mu.train1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess)
			totalGR = du.prod(growthRates)
			if i%10 == 0:
				#print(len(allActions))
				#print(allActions[4])
				print(i, 'th epoch')
				print('total growth rate:')
				print(totalGR)
				print()

if __name__=='__main__':
	main()



