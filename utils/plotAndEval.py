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
		self.returns = {} # a dictionary mapping each model's label to its return vector

	# Re should be a vector not shorter than Date. If Re is longer than date,
	# it will be automatically alinged to Date in terms of the last entry
	# label shoud be a string (e.g., 'cnn') 
	def addReturn(self, Re, label):
		self.returns[label] = Re[len(Re) - len(self.date):]


	def generatePlot(self):
		# Find the right place to put the year labels '20xx' 
		year = self.startYear + 1
		year_idx = {}
		for idx, date in enumerate(self.date):
			if year <= 2017:
				if int(date[0][0:4]) == year:
					year_idx[str(year)] = idx - 1
					year += 1
			else:
				break

		dateLabel = ['' for i in xrange(len(self.date))]
		for year in year_idx.keys():
			dateLabel[year_idx[year]] = year
		x = xrange(len(self.date))
		# Do the plotting
		# Plot weekly return
		legend_1 = []
		for label in self.returns.keys():
			plt.plot(x,self.returns[label],'-')
			legend_1.append(label)

		plt.xticks(x, dateLabel)
		plt.title('Weekly Return')
		plt.legend(legend_1)
		plt.xlabel('Year')
		plt.ylabel('Return')
		plt.savefig('results/Figures/Weekly_Re')
		plt.show()

		# plot (log) accumulated return
		legend_2 = []
		for label in self.returns.keys():
			accumReturn = accum(self.returns[label])
			plt.plot(x,accumReturn,'-')
			legend_2.append(label)

		plt.xticks(x, dateLabel)
		plt.title('Weekly Accumulated Log Return')
		plt.legend(legend_2)
		plt.xlabel('Year')
		plt.ylabel('Accumulated Log Return')
		plt.savefig('results/Figures/Weekly_Re_Accum_Log')
		plt.show()

	

	def eval(self):
		for label in self.returns.keys():
			print('Model: {0} \n total return: {1} \n sharpe ratio: {2}'
				.format(label, totalReturn(self.returns[label]), sharp_ratio(self.returns[label])))









