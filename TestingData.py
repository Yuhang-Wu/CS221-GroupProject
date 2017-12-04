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


## for baseline, can directly input (stockPrice,TestTimeIndex)
## for rnn and cnn, testing data is (TestReturn), and corresponding date is (TestDate)
def get_testing_data():
    DATA_PATH = 'data/sp10/'
    ## specify testing data
    # get data for testing
    dateSelected, stockPrice = du.getData(DATA_PATH, 'week')
    # get testing data index
    TestTimeIndex = range(len(dateSelected)/10*9, len(dateSelected))
    # get testing time period Date for estimating return (startDate, endDate)
    TestDate = [(dateSelected[i-1][0],dateSelected[i][0]) for i in TestTimeIndex]
    
    # get testing return
    stockReturn  = du.logReturn(stockPrice)
    TestReturn = [stockReturn[i-1] for i in TestTimeIndex]


    return TestDate, TestReturn, stockPrice, TestTimeIndex, dateSelected
