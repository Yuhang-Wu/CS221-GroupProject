import sys, os, csv
import numpy as np
from utils import dataUtil

def baseline(stockPrice, Time):
    # Time: time period index for baseline estimation  
    # estimate Return[i] = Return for Date period [Time[i],Time[i]+1]
    estimateReturn = np.zeros(len(Time)) 
    
    # total asset
    M = np.zeros(len(Time)+1) 
    M[0] = 1 
    
    # parameters for transaction cost
    c = np.zeros(len(stockPrice[-1])+1) + 0.0001
    
    # calculate return including transaction cost
    for i in xrange(len(Time)):
        # estimateReturn[i] is return for [Time[i], Time[i]+1]
        estimateReturn[i], eigenportfolio = calculateReturn(stockPrice[0:Time[i]+1]) 
        if i == 0:
            estimateReturn[i] -= c[0]+c[1]
            M[i+1] = M[i]*(1 + estimateReturn[i])
            beforePt = eigenportfolio
        else:
            flag = 0
            for j in xrange(len(stockPrice[-1])):
                tmp = eigenportfolio[j]-beforePt[j]*M[i-1]/M[i]*stockPrice[Time[i]][j]/stockPrice[Time[i]-1][j]
                estimateReturn[i] = estimateReturn[i] - c[j+1] * np.abs(tmp)
                if tmp != 0:
                    flag = 1
            if flag:
                estimateReturn[i] = estimateReturn[i] - c[0]
            M[i+1] = M[i]*(1 + estimateReturn[i])
            beforePt = eigenportfolio
    
    return estimateReturn
    

# stockReturn: return for each time period
def calculateReturn(stockPrice):
    
    """
    flag = 0
    for i in xrange(len(stPrice[-2])):
        if stPrice[-2][i] > stPrice[-3][i]:
            flag = 1
            break
    if flag == 0: # if stock prices all decrease in the previous time period, do not invest at current step
        return None,0,np.append([1],np.zeros(len(stPrice[-2])))
    """
    
    mu = xrange(10)
    stockReturn = getPeriodReturn(stockPrice)
    
    # choose mu_max and portReturn_max based on maximizing return for last second period
    mu_max = 0
    portReturn_max = -100000
    for mui in mu:
        portReturn, eigenportfolio = baselineReturn(stockReturn[:-1], mui)
        if portReturn > portReturn_max:
            portReturn_max = portReturn
            mu_max = mui
    return baselineReturn(stockReturn[:-1], mu_max)


def getPeriodReturn(stockPrice):
    stockReturn = np.empty((len(stockPrice)-1,len(stockPrice[0])))
    for i in xrange(len(stockPrice)-1):
        stockReturn[i] = (np.array(stockPrice[i+1]) - np.array(stockPrice[i]))/np.array(stockPrice[i])
    return stockReturn

    
def baselineReturn(stockReturn, mu):
    cov = getCovarianceMatrix(stockReturn[:-1], mu)
    eigenportfolio = getLargestEigenvector(cov)
    portReturn = np.dot(eigenportfolio, stockReturn[-1])
    return portReturn, eigenportfolio
    

# exponential weighted average mean and covariance matrix
# stocknum: num of stockPrice
# mu: hyperparameter for weighted exponential mean 
# T: number of time period
def getCovarianceMatrix(stockReturn, mu):
    stocknum = len(stockReturn[0])
    returnSum = np.zeros(stocknum)
    T = len(stockReturn)
    denumerator = 0
    
    # calculate weighted exponential mean
    for i in xrange(T):
        returnSum = returnSum + stockReturn[i] * np.exp(-mu * (T-i))
        denumerator += np.exp(-mu * (T-i))
    meanReturn = returnSum / denumerator

    # calculate weighted exponential covariance matrix
    cov = np.zeros((stocknum, stocknum))
    for i in xrange(T):
        normalizedReturn = (stockReturn[i] - meanReturn)
        cov = cov + \
        np.dot(np.transpose(normalizedReturn),normalizedReturn) * np.exp(-mu * (T-i))
    
    cov  = cov / denumerator
    return cov
        
    
# calculate largest eval-eigenvector 
def getLargestEigenvector(cov):
    D, S = np.linalg.eigh(cov)
    eigenportfolio = S[:,-1] / np.sum(S[:,-1]) 
    return eigenportfolio


"""
# get data (date, stockPrice)
dateSelected, stockPrice = dataUtil.getData()
    
# get time index for baseline estimation  
Time = range(len(dateSelected)/3,len(dateSelected)) 

# Date for estimated return period (startDate,endDate)
Date = [(dateSelected[i-2][0],dateSelected[i-1][0]) for i in Time]

# estimated period return for corresponding date
estimateReturn = baseline(stockPrice, Time)
"""


