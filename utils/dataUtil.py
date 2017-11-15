import numpy as np
import readin
# Select the dates which are the first date of each month in database
# store the result as a list of (date, index) tuple
def selectDate(allfilecontents, frequency = 'month'):
	if frequency == 'week':
		return selectDate_weekly(allfilecontents)
	curMonth = int(allfilecontents[0][1][1][0][5:7])
	dateSelected = []
	for dateIdx in xrange(len(allfilecontents[0][1][1:])):
		oneDay = allfilecontents[0][1][1:][dateIdx]
		if curMonth == int(oneDay[0][5:7]):
			dateSelected.append((oneDay[0], dateIdx))
			if curMonth == 12:
				curMonth = 1
			else:
				curMonth += 1
	return dateSelected[1:]


# Store the price of each stock on the date selected, return a matrix 
# in the form of [[]]
def getStockPrice(allfilecontents, dateSelected):
	stockPrice = [[] for i in xrange(len(dateSelected))]
	for compName, data in allfilecontents:
		for i in xrange(len( dateSelected)):
			dateIdx = dateSelected[i][1]
			price = float(data[1:][dateIdx][1]) #open price
			stockPrice[i].append(price)		
	return stockPrice


def loss2mRatio(loss):
	return 1.0 / (1.0 - loss)


# Select the dates which are the first date of each week in database
# store the result as a list of (date, index) tuple
def selectDate_weekly(allfilecontents):
	cws = allfilecontents[0][1][1][0] #the first day of current week 
	cws = cws[0:4] + cws[5:7] + cws[8:10]
	cws_int = int(cws) #covert string to integer
	dateSelected = []
	for dateIdx in xrange(len(allfilecontents[0][1][1:])):
		oneDay = allfilecontents[0][1][1:][dateIdx][0]
		oneDay_int = int(oneDay[0:4] + oneDay[5:7] + oneDay[8:10])
		if oneDay_int >= cws_int:
			dateSelected.append((oneDay, dateIdx))
			# update cws and cws_int
			if cws[4:6] =='12': #may need to drump to the next year
				if int(cws[6:8]) >= 25:
					cws = str(int(cws[0:4]) +1).zfill(4) + str('01') + str(int(cws[6:8])+7-31).zfill(2)
					cws_int = int(cws)
				else:
					cws_int += 7
					cws = str(cws_int)
			elif cws[4:6] in ['01', '03', '05', '07', '08', '10']:
				if int(cws[6:8]) >= 25:
					cws = str(int(cws[0:4])).zfill(4) + str(int(cws[4:6])+1).zfill(2) + str(int(cws[6:8])+7-31).zfill(2)
					cws_int = int(cws)
				else:
					cws_int += 7
					cws = str(cws_int)
			elif cws[4:6] == '02':
				if int(cws[0:4]) % 4 ==0: #run nian
					if int(cws[6:8]) >= 23:
						cws = str(int(cws[0:4])).zfill(4) + str('03') + str(int(cws[6:8])+7-29).zfill(2)
						cws_int = int(cws)
					else:
						cws_int += 7
						cws = str(cws_int)
				else: #pin nian
					if int(cws[6:8]) >= 22:
						cws = str(int(cws[0:4])).zfill(4) + str('03') + str(int(cws[6:8])+7-28).zfill(2)
						cws_int = int(cws)
					else:
						cws_int += 7
						cws = str(cws_int)
			else:
				if int(cws[6:8]) >= 24:
					cws = str(int(cws[0:4])).zfill(4) + str(int(cws[4:6])+1).zfill(2) + str(int(cws[6:8])+7-30).zfill(2)
					cws_int = int(cws)
				else:
					cws_int += 7
					cws = str(cws_int)
	return dateSelected[1:]

# get dates selected and stockprices
def getData(datapath1 = 'data/sp10/'):
	allfilecontents = readin.readCsvFromPath(datapath1)
	dateSelected = selectDate(allfilecontents, 'week')
	stockPrice =  getStockPrice(allfilecontents, dateSelected)
	dateSelected = dateSelected[1:]
	stockPrice = stockPrice[1:]
	return dateSelected, stockPrice
  
# get prices only
def getPrices(datapath1 = 'data/sp10/'):
	return np.array(getData(datapath1)[1])

def logReturn(stockPrice):
    logReturnPrices = np.log(np.array(stockPrice[1:])) - np.log(np.array(stockPrice[:-1]))
    return logReturnPrices

def logReturnMatrix(logReturnPrices, N):
    lRMtx =np.empty((len(logReturnPrices)-N,len(logReturnPrices[0]),N))
    for i in xrange(N,len(logReturnPrices)):
        lRMtx[i-N] = np.transpose(logReturnPrices[i-N:i])
    return lRMtx

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
    
    cov = cov / denumerator
    return cov
        
    
# calculate largest eval-eigenvector 
def getLargestEigenvector(cov):
    D, S = np.linalg.eigh(cov)
    eigenportfolio = S[:,-1] / np.sum(S[:,-1]) 
    return eigenportfolio

# portfolioHistory: a list/nparray of double
# indicating the total worth of portfolio
def calcMDD(portfolioHistory):
	ph = portfolioHistory
	mdd = 0.0
	for i in range(len(ph)):
		for j in range(i+1, len(ph)):
			mdd = max(mdd, (ph[i] - ph[j]) / ph[i])
	return mdd

def preprocess(stockPrice, N):
	prices = np.array(stockPrice)
	returnMatrix = np.empty((len(prices)-N,len(prices[0]),N))
	for i in range(len(prices) - N):
		returnMatrix[i] = np.transpose(prices[i:i+N]/prices[i+N-1])

	return np.log(returnMatrix)

def extendDim(arr):
	newshape = tuple(list(arr.shape) + [1])
	return np.reshape(arr, newshape)

def getInitialAllocation(D):
	prevA = np.array([0.0 for _ in range(D + 1)])
	prevA[-1] = 0.0
	return extendDim(prevA)

def prod(arr):
	p = 1.0
	for ele in arr:
		p *= ele
	return p

def getInputs(stockPrices, N, method = 'vsToday'):
	if method == 'vsToday':
		return getInputsVsToday(stockPrices, N)
	elif method == 'vsYesterday':
		return getInputsVsYesterday(stockPrices, N)

def getInputsVsYesterday(stockPrices, N):
	returnMatrix = logReturn(stockPrices)
	prevReturnMatrix = extendDim(returnMatrix[N-1:-1])
	nextReturnMatrix = extendDim(returnMatrix[N:])
	returnTensor = logReturnMatrix(returnMatrix, N)
	return returnTensor, prevReturnMatrix, nextReturnMatrix


def getInputsVsToday(stockPrices, N):
	returnMatrix = logReturn(stockPrices)
	prevReturnMatrix = extendDim(returnMatrix[N-2:-1])
	nextReturnMatrix = extendDim(returnMatrix[N-1:])
	returnTensor = preprocess(stockPrices, N)
	return returnTensor, prevReturnMatrix, nextReturnMatrix
