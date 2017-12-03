from __future__ import print_function
import numpy as np
import tensorflow as tf
from models import modelUtil as mu
from models import dummyModel as dm
from models import cnnModel as cm
from models import rnnModel as rm
from models import rnnModel2 as rm2
from utils import readin, yfReader, dataUtil as du
from utils import plotAndEval as pe
import EigenPortfolio as ep
import oracle as oc
import logging
import os
import bestStock as bs

DATA_PATH = 'data/sp10/'
DATA_PATH_ALL = 'data/sp150'
D = 10
N = 5
c = 0.0001
startFunding = 10000
epochs = 400
transCostParams = {
	'c': np.array([ [c] for _ in range(D) ]),
	'c0': c
}

resultsDirectory = 'results/allresults/' + du.getCurrentTimestamp() 
os.mkdir(resultsDirectory)

# now call logger.info to log
logger = du.setupLogger(resultsDirectory)

def main():
	print(du.getCurrentTimestamp())
	 
	trainAndTestTrial()
	#rnnModelTrainingTrial()

def cnnModelTrainingTrial():
	dateSelected, stockPriceList = du.getData(DATA_PATH, frequency = 'month')
	stockPrices = np.array(stockPriceList)

	D = stockPrices.shape[1]


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

	L = 4
	D = stockPrices.shape[1]
	
	# all the inputs!!
	returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrices, N, 'vsYesterday', L)

	# define model
	curModel = rm2.RnnModel(D, N, transCostParams, L = 4)
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

def trainAndTestTrial():
	dateSelected, trainPriceList, devPriceList, testPriceList = du.getTDTdata(DATA_PATH_ALL, frequency = 'week', getAll = True)
	
	baselineTime = range(10+(len(dateSelected)-10)/2+1,len(dateSelected)) 
	baselineTransCostParams = np.zeros(D + 1) + c

	# all the inputs!!
	epochs = 5
	L = 4
	# define model

	curModel = rm2.RnnModel(D, N, transCostParams, L = L)

	curModel.get_model_info()

	#quit()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for e in range(epochs):
			logger.info('Beginning '+str(e)+'_th epoch')
			logger.info('')
			for i in range(len(trainPriceList)):
				stockPrices = trainPriceList[i]
				returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrices, N, L = 4)
				allActions, growthRates = mu.train1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess)
				totalGR = du.prod(growthRates)
				
				logger.info(str(i) + '_th group in training')
				logger.info('total growth rate: '+ str(totalGR))
				logger.info('')
					
		for i in range(len(devPriceList)):
			stockPrices = devPriceList[i]
			returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrices, N, L=4)
			allActions, growthRates = mu.test1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess)
			
			baselineGrowthRates = 1.0 + ep.baseline(du.reduceDim(stockPrices), baselineTime, baselineTransCostParams)
			growthRates = growthRates[-len(baselineGrowthRates):]
			totalGR = du.prod(growthRates)
			baselineTotalGR = du.prod(baselineGrowthRates)
			logger.info(str(i) + '_th group in dev')
			logger.info('model total growth rate in dev: '+ str(totalGR))
			logger.info('baseline total growth rate: '+str(baselineTotalGR))
			logger.info('')
			


		for i in range(len(testPriceList)):
			stockPrices = testPriceList[i]
			returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrices, N, L=4)
			allActions, growthRates = mu.test1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess)
			baselineGrowthRates = 1.0 + ep.baseline(du.reduceDim(stockPrices), baselineTime, baselineTransCostParams)
			oracleGrowthRates = 1.0 + np.array(oc.solveOracle(du.reduceDim(stockPrices), startFunding, transCostParams['c'], transCostParams['c0']))
			#bestStockGrowthRates = 1.0 + np.array(bs.bestStock(du.reduceDim(stockPrices)))
			growthRates = growthRates[-len(baselineGrowthRates):]
			oracleGrowthRates = oracleGrowthRates[-len(baselineGrowthRates):]
			#bestStockGrowthRates = bestStockGrowthRates[-len(baselineGrowthRates):]
			totalGR = du.prod(growthRates)
			baselineTotalGR = du.prod(baselineGrowthRates)
			oracleTotalGR = du.prod(oracleGrowthRates)
			#bestStockTotalGR = du.prod(bestStockGrowthRates)
			logger.info(str(i) + '_th group in test')
			logger.info('model total growth rate in test: '+ str(totalGR))
			logger.info('baseline total growth rate: '+str(baselineTotalGR))
			logger.info('oracle total growth rate: '+str(oracleTotalGR))
			#logger.info('bestStock total growth rate: '+str(bestStockTotalGR))
			logger.info('')
			# plot
			plot_eval = pe.plotEval([dateSelected[i] for i in baselineTime], 2014)
			plot_eval.addReturn(growthRates, 'Model')
			plot_eval.addReturn(oracleGrowthRates, 'Oracle')
			#plot_eval.addReturn(bestStockGrowthRates, 'BestStock')
			plot_eval.addReturn(baselineGrowthRates, 'Baseline')
			plot_eval.generatePlot()

	
	   	

if __name__=='__main__':
	main()



