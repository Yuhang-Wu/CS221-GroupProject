import sys, os, csv
import numpy as np
import random
from utils import readin, yfReader, crawler

def yfReaderUsageExample():
	outpath = 'data/crawled'
	company = "AAPL"
	yfReader.getYfData(company,outpath)

def readCsvExample():
	datapath1 = 'data/sp10/'
	allfilecontents= readin.readCsvFromPath(datapath1)
	print allfilecontents[0]

'''
def crawlerUsageExample():
	datapath2 = 'data/crawled'
	company = "FB"
	crawler.crawl(company,datapath2)
'''
def get150():
	outpath = 'data/sp150'
	companyNamesFile = open('data/sp500tops.txt','r')
	companyCodes = companyNamesFile.readlines()
	print companyCodes
	for c in companyCodes:
		company = c.strip()
		if '.' in company:
			company = company.replace('.', '-')
		yfReader.getYfData(company,outpath)

def verify150():
	startDate = '2014-10-22'
	datapath1 = 'data/sp150/'
	allfilecontents= readin.readCsvFromPath(datapath1)
	print(len(allfilecontents))
	for c, f in allfilecontents:
		#print f[1][0]
		valid = f[1][0] == startDate
		if not valid: print c

def getTrainDevTest():
	outpath = 'data/'
	companyNamesFile = open('data/sp500tops.txt','r')
	companyCodes = companyNamesFile.readlines()
	companyCodes = [c.strip() for c in companyCodes]
	random.shuffle(companyCodes)
	tdtFile = open(outpath + 'train_dev_test.txt', 'w+')
	tdtFile.write('\n'.join(companyCodes))
	tdtFile.close()


def main():
	#getTrainDevTest()
	#testTFindex()
	#get150()
	#verify150()


if __name__=='__main__':
	main()