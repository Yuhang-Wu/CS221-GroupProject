import numpy as np

# Select the dates which are the first date of each month in database
# store the result as a list of (date, index) tuple
def selectDate(allfilecontents):
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
