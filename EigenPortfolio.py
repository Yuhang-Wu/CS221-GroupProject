import sys, os, csv
import numpy as np
from utils import dataUtil as du

def main():
    
    # get data
    dateSelected, stockPrice = du.getData()
    
    # get time for baseline estimation  
    # [0,10],[0,11],...,[0,len(dateSelected)]
    Time = xrange(10,len(dateSelected)+1) 
    Date = [dateSelected[i-2][0] for i in Time]
    
    estimateReturn = np.zeros(len(Time)) 
    
    # total asset
    M = np.zeros(len(Time)+1) 
    M[0] = 28348.5754100817
    
    # parameters for transaction cost
    c = np.zeros(len(stockPrice[-1])+1) + 0.001
    
    # calculate return including transaction cost
    for i in xrange(len(Time)):
        # estimateReturn[i] is return for [Time[i]-1, Time[i]]
        estimateReturn[i], eigenportfolio = calculateReturn(stockPrice[0:Time[i]]) 
        if i == 0:
            estimateReturn[i] -= 0.002
            M[i+1] = M[i]*(1 + estimateReturn[i])
            beforePt = eigenportfolio
        else:
            flag = 0
            for j in xrange(len(stockPrice[-1])):
                tmp = eigenportfolio[j]-beforePt[j]*M[i-1]/M[i]*stockPrice[Time[i]-1][j]/stockPrice[Time[i]-2][j]
                estimateReturn[i] = estimateReturn[i] - c[j+1] * np.abs(tmp)
                if tmp != 0:
                    flag = 1
            if flag:
                estimateReturn[i] = estimateReturn[i] - c[0]
            M[i+1] = M[i]*(1 + estimateReturn[i])
            beforePt = eigenportfolio
    
    return estimateReturn, Date, M
    

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
    cov = du.getCovarianceMatrix(stockReturn[:-1], mu)
    eigenportfolio = du.getLargestEigenvector(cov)
    portReturn = np.dot(eigenportfolio, stockReturn[-1])
    return portReturn, eigenportfolio
    



if __name__=='__main__':
    estimateReturn, Date, M = main()

