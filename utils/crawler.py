######## DEPRECATED ##########
#### use yfReader instead ####

# based on crawler implementation from github user Eroica-cpp
# https://github.com/Eroica-cpp/yahoo-finance-crawler.git
import urllib2
import BeautifulSoup
import re
import sys
import os

# date spans from 2014/10/22 to 2017/10/22
startDate = "1413961200"
endDate = "1508655600"
#urlPattern = ["https://finance.yahoo.com/quote/","/history?period1="+startDate+"&period2="+endDate+"&interval=1d&filter=history&frequency=1d"]
urlPattern=["http://query1.finance.yahoo.com/v7/finance/download/","?period1="+startDate+"&period2="+endDate+"&interval=1d&events=history&crumb=4pY5iACZrgf"]
def getHTML(URL, verbose = False):
	"""
	get raw HTML from given URL
	"""
	if verbose:
		print "downloading from " + URL
	req = urllib2.Request(URL)
	con = urllib2.urlopen(req)
	HTML = con.read()
	con.close()

	f = open('newway.html', "a")
	f.write(HTML)
	f.close()
	return HTML

def parse(HTML, filename, verbose = False):
	soup = BeautifulSoup.BeautifulSoup(HTML)
	pattern = re.compile(r"t[hd]") # pattern of every post in the collection page
	raw_results = soup.findAll("table", {"class": "W(100%) M(0)"})
	f = open(filename, "a")
	counter = 0
	print len(raw_results)
	for table in raw_results:
		lines = table.findAll("tr")
		write_content = ""
		if counter >= 1: lines = lines[1:]
		file_content = []
		for line in lines:
			items = line.findAll(pattern)
			file_content.append("\t".join([i.text for i in items]) + "\n")
		f.write(''.join(file_content[:-1] )+ '\n')
		counter += 1

	f.close()
	if verbose:
		print COMPANY + " DONE!"

def crawl(companycode, datapath, verbose = False):
	yahooUrl = urlPattern[0] + companycode + urlPattern[1]
	filename = os.path.join(datapath, companycode+'.csv')
	HTML = getHTML(yahooUrl, verbose)
	parse(HTML, filename, verbose)
	if verbose:
		print companycode + " saved to " + filename


def main():
	
	companycode= "FB"
	url =urlPattern[0] + companycode + urlPattern[1]
	HTML = getHTML(url, True)
	print HTML
	#parse(HTML)

if __name__ == "__main__":
	main()