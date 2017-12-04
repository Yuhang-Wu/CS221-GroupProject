from __future__ import print_function
import numpy as np
import tensorflow as tf
from models import modelUtil as mu
from models import dummyModel as dm
from models import cnnModel as cm
from models import rnnModel as rm
from models import rnnModel2 as rm2
from utils import plotAndEval
from utils import readin, yfReader, dataUtil as du, getPlots as gp
import EigenPortfolio as ep
import logging
import os
from utils import plotAndEval as pe

DATA_PATH = 'data/sp10/'
DATA_PATH_ALL = 'data/sp150'
D = 10
c = 0.0001
<<<<<<< HEAD
=======
epochs = 400
>>>>>>> 6b28f2aa0a7cf39b8b43dd38c302e5046d72d081
transCostParams = {
'c': np.array([ [c] for _ in range(D) ]),
'c0': c
}
baselineTransCostParams = np.zeros(D + 1) + c

# resultsDirectory = 'results/allresults/' + du.getCurrentTimestamp()
# os.mkdir(resultsDirectory)

# now call logger.info to log
# logger = du.setupLogger(resultsDirectory)


## specify testing data
# get data for testing
dateSelected, stockPrice = du.getData(DATA_PATH, 'week')
# get testing data index
TestTimeIndex = range(len(dateSelected)/10*9, len(dateSelected))
# get testing time period Date for estimating return (startDate, endDate)
TestDate = [(dateSelected[i-1][0],dateSelected[i][0]) for i in TestTimeIndex]

<<<<<<< HEAD
resultsDirectory = 'results/allresults/' + du.getCurrentTimestamp()
os.mkdir(resultsDirectory)
    
# now call logger.info to log
logger = du.setupLogger(resultsDirectory)
=======
>>>>>>> 6b28f2aa0a7cf39b8b43dd38c302e5046d72d081

# epochs and tolerance for training
epochs = 500
tol = 1e-7

def main():
<<<<<<< HEAD
    N = [1,5,10,15,20]
    kernelSize = [1,3,5]
    
    ModelGrowthRates = []
    ModeltotalGR = np.zeros((len(N),len(kernelSize)))
    maxReturn = -1000
    maxN = -1
    maxkernelSize = -1
    Test = 0
    for i in range(len(N)):
        for j in range(len(kernelSize)):
            print(du.getCurrentTimestamp())
            print(N[i], kernelSize[j], Test)
            growthRates, totalGR = TrainAndValidTrial(N[i], kernelSize[j], Test)
            ModeltotalGR[i][j] = totalGR
            if maxReturn < totalGR:
                maxReturn = totalGR
                maxN = N[i]
                maxkernelSize = kernelSize[j]

            # ModelGrowthRates.append(growthRates)
    Test = 1
    growthRates,totalGR = TrainAndValidTrial(maxN, logger, maxkernelSize, Test)
    print(ModeltotalGR)
    print('maxN is {}, maxkernelSize is {}'.format(maxN, maxkernelSize))
=======
    N = [15]
    kernelSize = [3]
    
    ModelGrowthRates = []
    maxReturn = -1000
    maxN = -1
    maxkernelSize = -1
    for Ni in N:
        for kernelSizei in kernelSize:
            print(du.getCurrentTimestamp())
            resultsDirectory = 'results/allresults/' + du.getCurrentTimestamp()
            os.mkdir(resultsDirectory)

            # now call logger.info to log
            logger = du.setupLogger(resultsDirectory)
            growthRates, totalGR = TrainAndValidTrial(Ni, logger, kernelSizei)
            if maxReturn < totalGR:
                maxReturn = totalGR
                maxN = Ni
                maxkernelSize = kernelSizei

            # ModelGrowthRates.append(growthRates)

    print ('maxN is {}, maxkernelSize is {}'.format(maxN, maxkernelSize))
>>>>>>> 6b28f2aa0a7cf39b8b43dd38c302e5046d72d081

    """
    baselineGrowthRates = 1.0 + ep.baseline(stockPrice, TestTimeIndex, baselineTransCostParams)
    baselineTotalGR = du.prod(baselineGrowthRates)
    HyperparameterPlot = plotAndEval.plotEval(TestDate, 2016)
    HyperparameterPlot.addReturn(baselineGrowthRates, 'baseline')
    for i in xrange(len(N)):
        HyperparameterPlot.addReturn(ModelGrowthRates[i], 'cnnModel, N = ' + str(N[i]))
    HyperparameterPlot.generatePlot()
    HyperparameterPlot.eval()
    """


<<<<<<< HEAD
def TrainAndValidTrial(N, kernelSize, Test):
=======
def TrainAndValidTrial(N, logger, kernelSize):
>>>>>>> 6b28f2aa0a7cf39b8b43dd38c302e5046d72d081
    
    # get testing data index
    TestIndex = [i-N-1 for i in TestTimeIndex]
    # get validation data index
    ValidationIndex = range(TestIndex[0]/9*8, TestIndex[0])
    # get training data index
    TrainIndex = range(TestIndex[0]/9*8)

    returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrice, N)


    ## get training data
    returnTensor_Train = np.array([returnTensor[_] for _ in TrainIndex])
    prevReturnMatrix_Train = np.array([prevReturnMatrix[_] for _ in TrainIndex])
    nextReturnMatrix_Train = np.array([nextReturnMatrix[_] for _ in TrainIndex])

    ## get validation data
    returnTensor_Valid = np.array([returnTensor[_] for _ in ValidationIndex])
    prevReturnMatrix_Valid = np.array([prevReturnMatrix[_] for _ in ValidationIndex])
    nextReturnMatrix_Valid = np.array([nextReturnMatrix[_] for _ in ValidationIndex])

    ## get testing data
    returnTensor_Test = np.array([returnTensor[_] for _ in TestIndex])
    prevReturnMatrix_Test = np.array([prevReturnMatrix[_] for _ in TestIndex])
<<<<<<< HEAD
    nextReturnMatrix_Test = np.array([nextReturnMatrix[_] for _ in TestIndex])
=======

    nextReturnMatrix_Test = np.array([nextReturnMatrix[_] for _ in TestIndex])    
    print(returnTensor_Test.shape)
   
    # generate xticks for plotting
    xticks = du.date2xtick(dateSelected[i] for i in TestTimeIndex)
>>>>>>> 6b28f2aa0a7cf39b8b43dd38c302e5046d72d081


    # epochs and tolerance for training
    logger.info('total epochs: '+str(epochs))
    logger.info('tolerance: '+str(tol))
    
    # define model
    L = 1
    curModel = cm.CnnModel(D, N, transCostParams, kernelSize, L)

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

        ## valid model
        allActions, growthRates = mu.test1epoch(returnTensor_Valid, prevReturnMatrix_Valid, nextReturnMatrix_Valid, curModel, sess)
        totalGR = du.prod(growthRates)
        logger.info('model total growth rate in validation data: '+ str(totalGR))
        
<<<<<<< HEAD
        ## test model
        if Test == 1:
            allActions, growthRates = mu.test1epoch(returnTensor_Valid, prevReturnMatrix_Valid, nextReturnMatrix_Valid, curModel, sess)
            totalGR = du.prod(growthRates)
            logger.info('model total growth rate in testing data: '+ str(totalGR))

        print(allActions[0])

=======
        print(allActions[0])


        # growthRates = growthRates[-len(baselineGrowthRates):]

>>>>>>> 6b28f2aa0a7cf39b8b43dd38c302e5046d72d081
    return growthRates, totalGR


if __name__=='__main__':
    main()

