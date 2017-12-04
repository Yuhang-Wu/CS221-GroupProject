import os
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()

def getYfData(companycode, datapath, startDate ="2014-10-22", endDate ="2017-10-22", verbose = False):
	if verbose:
		print("Getting data from Yahoo Finance")
		print("Company code:" + companycode)
		print()
	company = companycode
	data = pdr.get_data_yahoo(company, start=startDate, end=endDate)
	data.to_csv(os.path.join(datapath, company + '.csv'), encoding='utf-8')
	
