import sys, os, csv
import numpy as np
from utils import readin, yfReader
import cvxpy as cvx

def main():
	data = readCsvExample()
	dateSelected = selectDate(data)
	stockPrice =  getStockPrice(data, dateSelected)
	c = [0.0001]*10
	c_0 = 0.0001
	solveOracle(stockPrice, 10000, c, c_0)


def yfReaderUsageExample():
	outpath = 'data/crawled'
	company = "AAPL"
	yfReader.getYfData(company,outpath)

# Get stock price data from CSV files
def readCsvExample():
	datapath1 = 'data/sp10/'
	allfilecontents= readin.readCsvFromPath(datapath1)
	return allfilecontents

# Return the company names
def getNames(allfilecontents):
	pass


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

# The oracle senario
# Outputs:
# 1. M: Total asset over time 
# 2. P: Portfolio over time
# 3. R: Reward over time

def solveOracle(SP, startingFund,c,c_0):
	result_M = []
	result_M.append(startingFund)
	result_P = []
	result_R = []
	compNum = len(SP[0])
	ini_P_value = 1.0 / (compNum + 1)
	ini_P = [ini_P_value] * compNum
	result_P.append(ini_P)
	# calculate the return at the begining of the 2nd time period
	# ignore transaction cost for now
	R_1 = np.dot(np.divide(ini_P, SP[0]), np.subtract(SP[1],SP[0]))
	result_R.append(R_1)
	M_1 = startingFund * (1 + R_1)
	result_M.append(M_1)
	for i in xrange(1, (len(SP)-1)):
		P_pre = result_P[-1] #vector
		M_pre = result_M[-2]
		M = result_M[-1]
		S_pre = SP[i-1] #vector
		S = SP[i]
		S_next = SP[i+1]
		# Variables:
		P = cvx.Variable(compNum)
		# Constraints:
		constraints = [0<=P, P<=1, sum(P) <= 1]
		# Form Objective:
		'''
		print 'P:',P
		print 'M_p:', M_pre
		print 'P_p:',P_pre
		print 'M:',M
		print 'S_pre:', S_pre
		print 'S:', S
		print '1/S:', cvx.inv_pos(S)
		'''
		term1 = cvx.mul_elemwise(np.subtract(S_next, S), cvx.mul_elemwise(cvx.inv_pos(S), P))
		absTerm = cvx.abs(cvx.mul_elemwise(cvx.inv_pos(S), P) - cvx.mul_elemwise(  cvx.mul_elemwise(M_pre,P_pre), cvx.inv_pos( cvx.mul_elemwise(M,S_pre ))))
		term2 = cvx.mul_elemwise( S, cvx.mul_elemwise(c, absTerm))
		obj = cvx.Maximize(sum(term1 - term2) - c_0)
		# Form and solve problem:
		prob = cvx.Problem(obj, constraints)
		prob.solve()
		'''print prob.status'''
		result_P.append(P.value)
		R = prob.value
		result_R.append(R)
		M_next = M * (1+R)
		result_M.append(M_next)
	#print result_M
	#print len(result_M)
	#print result_P
	return_accum = [0] * len(result_R)
	for i in xrange(len(result_R)):
		for j in xrange(i + 1):
			return_accum[i] += result_R[j]
	print return_accum
	print result_R
	print result_M
 	#return return_accum
 
 
def makePlot(xAxis, yAxis):
	pass



if __name__=='__main__':
	main()

'''
def crawlerUsageExample():
	datapath2 = 'data/crawled'
	company = "FB"
	crawler.crawl(company,datapath2)
'''
