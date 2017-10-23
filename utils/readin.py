import csv
import sys
import os
	
	# read in one single csv file
def readCsvFile(filename):
	f = open(filename, 'rb')
	reader = csv.reader(f)
	out = []
	
	for line in reader:
		out.append(line)
	f.close()
	return out

	# read in all the csv files and 
	# return them in a list of tuples
	# of the form (companyCode, data)
def readCsvFromPath(datapath):
	print "extracting csv data from "+datapath
	datafiles = [file for file in os.listdir(datapath) if file.endswith('.csv')]
	print datafiles
	output = []
	for file in datafiles:
		filename = os.path.join(datapath, file)
		filecontent = readCsvFile(filename)
		output.append((filename.split('.')[0], filecontent))
	return output

def separateDates(dateStr):
	return dateStr.split('-')



