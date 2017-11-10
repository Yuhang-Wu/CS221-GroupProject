import sys, os, csv
import numpy as np
from utils import readin

# getData:
def getData():
    data = readCsvExample()
    dateSelected = selectDate(data)
    stockPrice =  getStockPrice(data, dateSelected)
    dateSelected = dateSelected[1:]
    stockPrice = stockPrice[1:]
    return dateSelected, stockPrice
    
    
    
# Get stock price data from CSV files
def readCsvExample():
    allfilecontents= readin.readCsvFromPath('data/sp10/')
    return allfilecontents


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
    return dateSelected

# Store the price of each stock on the date selected, return a matrix 
# in the form of [[],]
def getStockPrice(allfilecontents, dateSelected):
    stockPrice = [[] for i in xrange(len(dateSelected))]
    for compName, data in allfilecontents:
        for i in xrange(len( dateSelected)):
            dateIdx = dateSelected[i][1]
            price = float(data[1:][dateIdx][1]) #open price
            stockPrice[i].append(price)
    return stockPrice


