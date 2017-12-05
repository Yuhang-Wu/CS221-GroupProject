from __future__ import print_function
import math
import matplotlib.pyplot as plt
import numpy as np

def accum(Re):
		result = []
		result.append(math.log(Re[0]))
		for idx in xrange(1, len(Re)):
			cur = result[-1] + math.log(Re[idx])
			result.append(cur)
		return result

# calculate total return and sharp ration
def totalReturn(Re):
	result = 1
	for re in Re:
		result *= re		
	return result

def sharp_ratio(Re):
	Re = np.array(Re) -1 
	return np.mean(Re)/np.std(Re)

class plotEval:
	def __init__(self, date, startYear):
		self.date = date
		self.startYear = startYear # an integer like 2015
		self.returns = []

	# Re should be a vector not shorter than Date. If Re is longer than date,
	# it will be automatically alinged to Date in terms of the last entry
	# label shoud be a string (e.g., 'cnn') 
	def addReturn(self, Re, label):
		self.returns.append((label,Re[len(Re) - len(self.date):]))


	def generatePlot(self):
		# Find the right place to put the year labels '20xx'
		# Show some of the dates as xticks (every 10)
		year = int(self.date[0][0][0:4])
		year_idx = {}
		date_idx = {}
		for idx, date in enumerate(self.date):
			if idx % 5 == 1:
				date_idx[date[0][2:]] = idx 
			if year <= 2017:
				if int(date[0][0:4]) == year:
					year_idx[str(year)] = idx 
					year += 1
			
		dateLabel = ['' for i in xrange(len(self.date))]
		
		for date_i in date_idx.keys():
			dateLabel[date_idx[date_i]] = date_i
		'''
		for year in year_idx.keys():
			dateLabel[year_idx[year]] = year
		'''
		x = xrange(len(self.date))
		# Do the plotting
		# Plot weekly return
		style = ['r-', 'g-', 'b-', 'c-', 'm-', 'y-', 'k-']
		legend_1 = []
		style_i = 0
		for (label_i, Re) in self.returns:
			plt.plot(x,Re,style[style_i], label = label_i)
			style_i += 1

		plt.xticks(x, dateLabel)
		plt.title('Weekly Return')
		plt.legend(loc = 'upper right')
		plt.xlabel('Date')
		plt.ylabel('Return')
		plt.savefig('results/Figures/Weekly_Re')
		plt.show()

		# plot (log) accumulated return
		legend_2 = []
		style_i = 0
		for (label_i, Re) in self.returns:
			accumReturn = accum(Re)
			plt.plot(x,accumReturn,style[style_i], label = label_i)
			style_i += 1

		plt.xticks(x, dateLabel)
		plt.title('Weekly Accumulated Log Return')
		plt.legend(loc = 'upper left')
		plt.xlabel('Date')
		plt.ylabel('Accumulated Log Return')
		plt.savefig('results/Figures/Weekly_Re_Accum_Log')
		plt.show()

	

	def eval(self):
		for label in self.returns.keys():
			print('Model: {0} \n total return: {1} \n sharpe ratio: {2}'
				.format(label, totalReturn(self.returns[label]), sharp_ratio(self.returns[label])))









