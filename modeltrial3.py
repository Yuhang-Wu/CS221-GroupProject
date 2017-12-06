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
epochs = 400
transCostParams = {
    'c': np.array([ [c] for _ in range(D) ]),
    'c0': c
}
baselineTransCostParams = np.zeros(D + 1) + c

# resultsDirectory = 'results/allresults/' + du.getCurrentTimestamp()
# os.mkdir(resultsDirectory)

# now call logger.info to log
# logger = du.setupLogger(resultsDirectory)
resultsDirectory = 'results/allresults/' + du.getCurrentTimestamp() 
os.mkdir(resultsDirectory)
logger = du.setupLogger(resultsDirectory)
logger.info('results saved to '+ resultsDirectory)
## specify testing data
# get data for testing
dateSelected, stockPrice = du.getData(DATA_PATH, 'week', getAll = True)
# get testing data index
TestTimeIndex = range(len(dateSelected)/10*9, len(dateSelected))
'''
TestIndex = [i-N-1 for i in TestTimeIndex]
ValidationIndex = range(TestIndex[0]/9*8, TestIndex[0])
TestDate = [(dateSelected[i-1][0],dateSelected[i][0]) for i in TestTimeIndex]
TrainIndex = range(TestIndex[0]/9*8)
'''
L = 4
# epochs and tolerance for training
epochs = 25
tol = 1e-7
def main():
    varyHS()
    #varyN()

def varyN():
    Ns = [5, 7, 10, 12, 15]
    HSs = [10]
    devGRs = []
    testGRs = []
    for N in Ns:
        logger.info('logging with N = ' + str(N))
        with tf.variable_scope('var_scope'+str(N)):
            devGR, testGR = TrainAndValidTrial(N, logger, HSs[0])
            devGRs.append(devGR)
            testGRs.append(testGR)

    
    devGRs = du.enforceMinlen(devGRs)
    testGRs = du.enforceMinlen(testGRs)

    logger.info('devGRs')
    logger.info(devGRs)
    logger.info('testGRs')
    logger.info(testGRs)
    '''
    for i in range(len(Ns)):
        print(len(devGRs[i]))
    for i in range(len(Ns)):
        print(len(testGRs[i]))
    #quit()
    '''

    devXticks = [str(i+1) for i in range(len(devGRs[0]))]
    testXticks = [str(i+1) for i in range(len(testGRs[0]))]

    devPlotter1 = gp.Plotter('dev-N', devXticks, 'week', 'return')
    for i in range(len(Ns)):
        devPlotter1.addLine(devGRs[i], 'N = '+str(Ns[i]))
    devPlotter1.plot(resultsDirectory + '/dev_vary_N.png')

    devPlotter2 = gp.Plotter('dev-N-accum', devXticks, 'week', 'accumulated return')
    for i in range(len(Ns)):
        devPlotter2.addLine(du.getAccumulatedReturn(devGRs[i]), 'N = '+str(Ns[i]))
    devPlotter2.plot(resultsDirectory + '/dev_vary_N_accum.png')

    testPlotter1 = gp.Plotter('test-N', testXticks, 'week', 'return')
    for i in range(len(Ns)):
        testPlotter1.addLine(testGRs[i], 'N = '+str(Ns[i]))
    testPlotter1.plot(resultsDirectory + '/test_vary_N.png')

    testPlotter2 = gp.Plotter('test-N-accum', testXticks, 'week', 'return')
    for i in range(len(Ns)):
        testPlotter2.addLine(du.getAccumulatedReturn(testGRs[i]), 'N = '+str(Ns[i]))
    testPlotter2.plot(resultsDirectory + '/test_vary_N_accum.png')
    #print ('maxN is {}, maxkernelSize is {}'.format(maxN, maxkernelSize))

def varyHS():
    HSs = [5, 7, 10, 12, 15]
    Ns = [10]
    devGRs = []
    testGRs = []
    N = 10
    for HS in HSs:
        logger.info('logging with HS = ' + str(HS))
        logger.info('')
        with tf.variable_scope('var_scope'+str(HS)):
            devGR, testGR = TrainAndValidTrial(Ns[0], logger, HS)
            devGRs.append(devGR)
            testGRs.append(testGR)

    
    devGRs = du.enforceMinlen(devGRs)
    testGRs = du.enforceMinlen(testGRs)

    logger.info('devGRs')
    logger.info(devGRs)
    logger.info('testGRs')
    logger.info(testGRs)
    '''
    for i in range(len(Ns)):
        print(len(devGRs[i]))
    for i in range(len(Ns)):
        print(len(testGRs[i]))
    #quit()
    '''

    devXticks = [str(i+1) for i in range(len(devGRs[0]))]
    testXticks = [str(i+1) for i in range(len(testGRs[0]))]

    devPlotter1 = gp.Plotter('dev-HS', devXticks, 'week', 'return')
    for i in range(len(HSs)):
        devPlotter1.addLine(devGRs[i], 'HS = '+str(HSs[i]))
    devPlotter1.plot(resultsDirectory + '/dev_vary_HS.png')

    devPlotter2 = gp.Plotter('dev-HS-accum', devXticks, 'week', 'accumulated return')
    for i in range(len(HSs)):
        devPlotter2.addLine(du.getAccumulatedReturn(devGRs[i]), 'HS = '+str(HSs[i]))
    devPlotter2.plot(resultsDirectory + '/dev_vary_HS_accum.png')

    testPlotter1 = gp.Plotter('test-HS', testXticks, 'week', 'return')
    for i in range(len(HSs)):
        testPlotter1.addLine(testGRs[i], 'HS = '+str(HSs[i]))
    testPlotter1.plot(resultsDirectory + '/test_vary_HS.png')

    testPlotter2 = gp.Plotter('test-HS-accum', testXticks, 'week', 'return')
    for i in range(len(HSs)):
        testPlotter2.addLine(du.getAccumulatedReturn(testGRs[i]), 'HS = '+str(HSs[i]))
    testPlotter2.plot(resultsDirectory + '/test_vary_HS_accum.png')
    #print ('maxN is {}, maxkernelSize is {}'.format(maxN, maxkernelSize))

def TrainAndValidTrial(N, logger, hiddenSize):
    
    # get testing data index
    TestIndex = [i-N-1 for i in TestTimeIndex]
    # get validation data index
    ValidationIndex = range(TestIndex[0]/9*8, TestIndex[0])
    # get training data index
    TrainIndex = range(TestIndex[0]/9*8)

    returnTensor, prevReturnMatrix, nextReturnMatrix = du.getInputs(stockPrice, N, L = L)

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

    nextReturnMatrix_Test = np.array([nextReturnMatrix[_] for _ in TestIndex])    
    #print(returnTensor_Test.shape)
   
    # generate xticks for plotting
    xticks = du.date2xtick(dateSelected[i] for i in TestTimeIndex)


    # epochs and tolerance for training
    logger.info('total epochs: '+str(epochs))
    logger.info('tolerance: '+str(tol))
    
    # define model
    curModel = rm2.RnnModel(D, N, transCostParams, L = L, hiddenSize = hiddenSize)

    model_info = curModel.get_model_info()
    logger.info('model basic config')
    logger.info(model_info)

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
        devGrowthRates = growthRates
        logger.info('dev growth rate: '+ str(du.prod(growthRates)))
        
        allActions, growthRates = mu.test1epoch(returnTensor_Test, prevReturnMatrix_Test, nextReturnMatrix_Test, curModel, sess)
        testGrowthRates = growthRates
        logger.info('test growth rate: '+ str(du.prod(growthRates)))

        #print(allActions[0])
        # growthRates = growthRates[-len(baselineGrowthRates):]

    return devGrowthRates, testGrowthRates


if __name__=='__main__':
    main()

