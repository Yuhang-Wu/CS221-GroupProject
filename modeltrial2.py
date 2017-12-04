from __future__ import print_function
import numpy as np
import tensorflow as tf
from models import modelUtil as mu
from models import dummyModel as dm
from models import cnnModel as cm
from models import rnnModel as rm
from models import rnnModel2 as rm2
from utils import readin, yfReader, dataUtil as du, getPlots as gp
import EigenPortfolio as ep
import logging
import os
from utils import plotAndEval as pe
import oracle as oc

DATA_PATH = 'data/sp10/'
DATA_PATH_ALL = 'data/sp150'
D = 10
N = 10
c = 0.0001
epochs = 20
transCostParams = {
'c': np.array([ [c] for _ in range(D) ]),
'c0': c
}
baselineTransCostParams = np.zeros(D + 1) + c

resultsDirectory = 'results/allresults/' + du.getCurrentTimestamp() 
os.mkdir(resultsDirectory)

# now call logger.info to log
logger = du.setupLogger(resultsDirectory)

def main():
    print(du.getCurrentTimestamp())

    trainAndTestTrial()
    #rnnModelTrainingTrial()


def trainAndTestTrial():
    ## specify testing data
    # get data for testing
    dateSelected, stockPrice = du.getData(DATA_PATH, 'week')
    # get testing data index
    TestTimeIndex = range(len(dateSelected)/2, len(dateSelected))
    # get testing time period Date for estimating return (startDate, endDate)
    TestDate = [(dateSelected[i-1][0],dateSelected[i][0]) for i in TestTimeIndex]

    # get testing data index
    TestIndex = [i-N-1 for i in TestTimeIndex]
    # get training data index
    TrainIndex = range(TestIndex[0]) 

    returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrice, N)
    print(returnTensor.shape)
 
    ## get training data
    returnTensor_Train = np.array([returnTensor[_] for _ in TrainIndex])
    prevReturnMatrix_Train = np.array([prevReturnMatrix[_] for _ in TrainIndex])
    nextReturnMatrix_Train = np.array([nextReturnMatrix[_] for _ in TrainIndex])
    print(returnTensor_Train.shape)
    ## get testing data
    returnTensor_Test = np.array([returnTensor[_] for _ in TestIndex])
    prevReturnMatrix_Test = np.array([prevReturnMatrix[_] for _ in TestIndex])
    nextReturnMatrix_Test = np.array([nextReturnMatrix[_] for _ in TestIndex])    
    
    # generate xticks for plotting
    xticks = du.date2xtick(dateSelected[i] for i in TestTimeIndex)

    # epochs and tolerance for training 
    epochs = 500
    logger.info('total epochs: '+str(epochs))
    tol = 1e-7
    logger.info('tolerance: '+str(tol))
    
    # define model
    curModel = cm.CnnModel(D, N, transCostParams)
    L = 1

    model_info = curModel.get_model_info()
    logger.info('model basic config')
    logger.info(model_info)

    #quit()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## train model
        preTotalGR = 0
        for e in range(epochs):
            logger.info('Beginning '+str(e)+'_th epoch')
            logger.info('')

            allActions, growthRates = mu.train1epoch(returnTensor_Train, prevReturnMatrix_Train, nextReturnMatrix_Train, curModel, sess)
            totalGR = du.prod(growthRates)

            logger.info('model total growth rate in training data: '+ str(totalGR))
            logger.info('')
            
            if np.abs(preTotalGR - totalGR) < tol:
                break
            else:
                preTotalGR = totalGR

        ## test model
        allActions, growthRates = mu.test1epoch(returnTensor_Test, prevReturnMatrix_Test, nextReturnMatrix_Test, curModel, sess)
        print(allActions[0])

        baselineGrowthRates = 1.0 + ep.baseline(stockPrice, TestTimeIndex, baselineTransCostParams)
        oracleGrowthRates = oc.solveOracle(stockPrice, 10000, transCostParams['c'], transCostParams['c0'])
        growthRates = growthRates[-len(baselineGrowthRates):]
        totalGR = du.prod(growthRates)
        baselineTotalGR = du.prod(baselineGrowthRates)
        plotE = pe.plotEval(TestDate, 2016)
        plotE.addReturn(baselineGrowthRates, 'Baseline')
        plotE.addReturn(growthRates, 'CNN')
        plotE.addReturn(oracleGrowthRates, 'Oracle')
        plotE.generatePlot()
        logger.info('model total growth rate in testing data: '+ str(totalGR))
        logger.info('baseline total growth rate: '+str(baselineTotalGR))
        logger.info('')

if __name__=='__main__':
    main()