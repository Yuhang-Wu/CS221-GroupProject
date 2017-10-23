import sys, os, csv
import numpy as np
from utils import readin, yfReader

def main():
	yfReaderUsageExample()

def yfReaderUsageExample():
	outpath = 'data/crawled'
	company = "FB"
	yfReader.getYfData(company,outpath)

def readCsvExample():
	datapath1 = 'data/sp10/'
	allfilecontents= readin.readCsvFromPath(datapath1)
	print allfilecontents

if __name__=='__main__':
	main()


'''
def crawlerUsageExample():
	datapath2 = 'data/crawled'
	company = "FB"
	crawler.crawl(company,datapath2)
'''
