import numpy as np
import readin
import time, datetime
import logging
import os

# get accumulated return based on growth rates (a list of floats like 1.02)
# e.g. [1.2, 1.0, 1.5] -> [1.2, 1.2, 1.8]
def getAccumulatedReturn(growthRates):
	cur = growthRates[0]
	out = [cur]
	for i in range(1, len(growthRates)):
		cur *= growthRates[i]
		out.append(cur)
	return out

# get the year of a date string of format 'yyyy-mm-dd'
def getYear(date):
	return int(date[0][:4])

# get the date x ticks 
def date2xtick(dateSelected):
	s = set([])
	ticks = []
	for date in dateSelected:
		y = getYear(date)
		if y in s:
			ticks.append('')
		else:
			s.add(y)
			ticks.append(y)
	ticks[0] = ''
	return ticks

def setupLogger(outPath):
	# get the output file name
	logFileName = outPath + '/model_log.txt'

	# create logger
	logger = logging.getLogger('drl_in_pm')

	# create formatter
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

	# get the file handler
	hdlr = logging.FileHandler(logFileName)
	hdlr.setFormatter(formatter)

	# get the stream handler for system stdout
	sh = logging.StreamHandler()
	sh.setFormatter(formatter)

	# add the handlers
	logger.addHandler(hdlr) 
	logger.addHandler(sh)

	# set level to debug
	logger.setLevel(logging.DEBUG)

	return logger

# get the current timestamp
# return it as a string
def getCurrentTimestamp():
	ts = time.time()
	formattedTime = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
	return formattedTime

# take the last price in the last dimension (for rnn only)
def reduceDim(stockPricesAll):
	return stockPricesAll[:,:,-1]

# get all the data (train dev test)
# tdt stands for train dev test
# return dates, list of train prices, list of dev prices, list of test prices
def getTDTdata(datapath = 'data/sp150/', frequency = 'week', getAll  = False):
	allfilecontents = readin.readCsvFromPath(datapath)
	dateSelected = selectDate(allfilecontents, frequency)
	tdtCompanies = getTrainDevTestCompanies()
	fcs = divideFileContents(allfilecontents, tdtCompanies[0], tdtCompanies[1], tdtCompanies[2])
	fcLists = [divideToLists(fc) for fc in fcs]
	priceLists = [[getPricesBasedOnDates(ele, dateSelected, getAll) for ele in fcList] for fcList in fcLists]
	dateSelected = dateSelected[1:]
	return dateSelected, priceLists[0], priceLists[1], priceLists[2]

# get dates selected and stockprices
def getData(datapath1 = 'data/sp10/', frequency = 'week', getAll = False):
	allfilecontents = readin.readCsvFromPath(datapath1)
	dateSelected = selectDate(allfilecontents, frequency)
	stockPrice = getPricesBasedOnDates(allfilecontents, dateSelected, getAll)
	dateSelected = dateSelected[1:]
	return dateSelected, stockPrice

# pretty self explanatory, if getAll is true, get all the data (all 4 dimensions)
def getPricesBasedOnDates(allfilecontents, dateSelected, getAll):
	if getAll:
		stockPrice = getAllStockPrice(allfilecontents, dateSelected)
	else:
		stockPrice = getStockPrice(allfilecontents, dateSelected)
	return np.array(stockPrice[1:])

# get the names of all the companies in train dev and test set
# return a list of list of strings
def getTrainDevTestCompanies(filename = 'data/train_dev_test.txt', division= [110, 20, 20]):
	allCompanies = getLines(filename)
	#print allCompanies
	out = []
	out.append(allCompanies[:division[0]])
	out.append(allCompanies[division[0]: -division[2]])
	out.append(allCompanies[-division[2]:])
	return out

# given a file name, get all the companies stored in the file
def getLines(filename):
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
	return [ele.strip().replace('.','-') for ele in lines]

# divide file contentList into list of fileContent in units of D
def divideToLists(fileContentsList, D = 10):
	assert(len(fileContentsList)%D == 0, "not divisible, check fileContentList length")
	out = []
	for i in range(len(fileContentsList)/D):
		out.append(fileContentsList[i*D : i*D + D])
	return out

# divide them up based on what group they are in
def divideFileContents(fileContents, trainCompanies, devCompanies, testCompanies):
	d = {fileContents[i][0].split('/')[-1]:i for i in range(len(fileContents))}
	#print d
	trainFileContents = [fileContents[d[c]] for c in trainCompanies]
	devFileContents = [fileContents[d[c]] for c in devCompanies]
	testFileContents = [fileContents[d[c]] for c in testCompanies]
	return [trainFileContents, devFileContents, testFileContents]
	
# Select the dates which are the first date of each month in database
# store the result as a list of (date, index) tuple
def selectDate(allfilecontents, frequency = 'month'):
	if frequency == 'week':
		return selectDateWeekly(allfilecontents)
	elif frequency == 'day':
		return selectDateDaily(allfilecontents)
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
	allStockPrice = getAllStockPrice(allfilecontents, dateSelected)
	return getStockPriceFromAll(allStockPrice)

def getStockPriceFromAll(allStockPrice):
	stockPrice = [[] for i in xrange(len(allStockPrice))]
	for i in range(len(allStockPrice)):
		# get adj close price
		stockPrice[i]=[ele[-1] for ele in allStockPrice[i]] 
	return np.array(stockPrice)

def getAllStockPrice(allfilecontents, dateSelected):
	allStockPrice = [[] for i in xrange(len(dateSelected))]
	for compName, data in allfilecontents:
		for i in xrange(len( dateSelected)):
			dateIdx = dateSelected[i][1]
			prices = [float(data[1:][dateIdx][j]) for j in range(1,5)] #open price
			allStockPrice[i].append(prices)		
	return allStockPrice

def loss2mRatio(loss):
	return 1.0 / (1.0 - loss)

def selectDateDaily(allfilecontents):
	#dateSelected = []
	return [(0,i) for i in range(len(allfilecontents[0][1]) - 1) ]

# Select the dates which are the first date of each week in database
# store the result as a list of (date, index) tuple
def selectDateWeekly(allfilecontents):
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

<<<<<<< HEAD
# getData:
def getData(datapath1 = 'data/sp10/'):
	allfilecontents = readin.readCsvFromPath(datapath1)
	dateSelected = selectDate(allfilecontents, 'week')
	stockPrice =  getStockPrice(allfilecontents, dateSelected)
	dateSelected = dateSelected[1:]
	stockPrice = stockPrice[1:]
	return dateSelected, stockPrice
    
=======
  
# get prices only
def getPrices(datapath1 = 'data/sp10/'):
	return np.array(getData(datapath1)[1])

>>>>>>> 6b28f2aa0a7cf39b8b43dd38c302e5046d72d081
def logReturn(stockPrice):
    logReturnPrices = np.log(np.array(stockPrice[1:])) - np.log(np.array(stockPrice[:-1]))
    return logReturnPrices

def logReturnMatrix(logReturnPrices, N, L = 1):
	#print L
	if L == 1:
		lRMtx = np.empty((len(logReturnPrices) - N, len(logReturnPrices[0]), N))
	else:
		lRMtx = np.empty((len(logReturnPrices) - N, len(logReturnPrices[0]), N, L))
	for i in xrange(N, len(logReturnPrices)):
		if L == 1:
			lRMtx[i-N] = np.transpose(logReturnPrices[i - N : i])
		else:
			lRMtx[i-N] = np.transpose(logReturnPrices[i - N : i], (1, 0, 2))
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

def preprocess(stockPrice, N, L = 1):
	prices = np.array(stockPrice)
	if L == 1:
		returnMatrix = np.empty((len(prices)-N,len(prices[0]),N))
	else:
		returnMatrix = np.empty((len(prices)-N,len(prices[0]),N,L))
	for i in range(len(prices) - N):
		if L == 1:
			curPrices = prices[i+N-1,:]
			divisor = curPrices
			returnMatrix[i] = np.transpose(prices[i:i+N]/divisor)
		else:
			curPrices = prices[i+N-1,:,-1]
			divisor = np.stack([curPrices for _ in range(L)], axis = 2)
			returnMatrix[i] = np.transpose(prices[i:i+N]/divisor, (1,0,2))
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

def getInputs(stockPrices, N, method = 'vsYesterday', L = 1):
	if method == 'vsToday':
		return getInputsVsToday(stockPrices, N, L)
	elif method == 'vsYesterday':
		return getInputsVsYesterday(stockPrices, N, L)

def getInputsVsYesterday(stockPrices, N, L = 1):
	#print L
	returnMatrix = logReturn(stockPrices)
	#print(returnMatrix.shape)
	if L == 1:
		prevReturnMatrix = extendDim(returnMatrix[N-1:-1])
		nextReturnMatrix = extendDim(returnMatrix[N:])
	else:
		stockPricesLast = getStockPriceFromAll(returnMatrix)
		prevReturnMatrix = extendDim(stockPricesLast[N-1:-1])
		nextReturnMatrix = extendDim(stockPricesLast[N:])
	returnTensor = logReturnMatrix(returnMatrix, N, L)
	return returnTensor, prevReturnMatrix, nextReturnMatrix


def getInputsVsToday(stockPrices, N, L = 1):
	returnMatrix = logReturn(stockPrices)
	if L == 1:
		prevReturnMatrix = extendDim(returnMatrix[N-2:-1])
		nextReturnMatrix = extendDim(returnMatrix[N-1:])
	else:
		stockPricesLast = getStockPriceFromAll(returnMatrix)
		prevReturnMatrix = extendDim(stockPricesLast[N-2:-1])
		nextReturnMatrix = extendDim(stockPricesLast[N-1:])
	returnTensor = preprocess(stockPrices, N, L)
	return returnTensor, prevReturnMatrix, nextReturnMatrix
