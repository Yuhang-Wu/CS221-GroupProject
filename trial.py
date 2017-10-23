import sys, os, csv
import numpy as np
from utils import readin

def main():
	datapath = 'data/sp10/'
	allfilecontents= readin.readCsvFromPath(datapath)
	print allfilecontents


if __name__=='__main__':
	main()

