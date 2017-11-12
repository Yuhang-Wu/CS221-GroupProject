import sys, os, csv
import numpy as np
from utils import readin, yfReader

def main():
	readCsvExample()

def yfReaderUsageExample():
	outpath = 'data/crawled'
	company = "AAPL"
	yfReader.getYfData(company,outpath)

def readCsvExample():
	datapath1 = 'data/sp10/'
	allfilecontents= readin.readCsvFromPath(datapath1)
	print allfilecontents[0]

if __name__=='__main__':
	main()


'''
def crawlerUsageExample():
	datapath2 = 'data/crawled'
	company = "FB"
	crawler.crawl(company,datapath2)
'''
