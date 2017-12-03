import math
import matplotlib.pyplot as plt

# please stop writing imperative
# and unmodularized mode please
class Plotter(object):
	def __init__(self, title, dates, xlabel, ylabel):
		self.title = title
		self.dates = dates
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.lines = []
		self.legends = []
		self.options = []

	def addLine(self, line, legend, option = '-'):
		assert(len(line) == len(self.dates), "plotter dimensionality mismatch")
		self.lines.append(line)
		self.legends.append(legend)
		self.options.append(option)

	def plot(self, outPath):
		#self.enforceMinLen()
		plt.figure()
		x = xrange(len(self.dates))
		for i in range(len(self.lines)):
			plt.plot(x, self.lines[i], self.options[i])
		plt.xticks(x, self.dates)
		plt.title(self.title)
		plt.legend(self.legends)
		plt.xlabel(self.xlabel)
		plt.ylabel(self.ylabel)
		plt.savefig(outPath)
		#plt.show()

	def getMinLen(self):
		minlen = len(self.dates)
		for line in self.lines:
			minlen = min(minlen, len(line))
		return minlen

	def enforceMinLen(self):
		minlen = self.getMinLen()
		self.dates = self.dates[ -minlen: ]
		for i in range(len(self.lines)):
			self.lines[i] = self.lines[i][ -minlen: ]

# get the accumulated return for each senario
def accum(Re):
	result = []
	result.append(Re[0])
	for idx in xrange(1, len(Re)):
		cur = result[-1] + Re[idx]
		result.append(cur)
	return result

# Each Re_xx is the (test) return vector for each senario: bl is baseline, oc is oracle, rnn is rnn
# cnn is cnn. Date shoud be have the form [('2016-05-25', '2016-06-01'), ('2016-06-01', '2016-06-08'), ...]
# startYear is an integer, eg., 2014
def plot(Re_bl, Re_oc, Re_rnn, Re_cnn, Date, startYear):

	# Find the right place to put the year labels '20xx' 
	year = startYear + 1
	year_idx = {}
	for idx, date in enumerate(Date):
		if year <= 2017:
			if int(date[0][0:4]) == year:
				year_idx[str(year)] = idx
				year += 1
		else:
			break

	dateLabel = ['' for i in xrange(len(Date))]
	for year in year_idx.keys():
		dateLabel[year_idx[year]] = year
	x = xrange(len(Date))

	# get accumulated return
	Re_bl_accum = accum(Re_bl)
	Re_oc_accum = accum(Re_oc)
	Re_cnn_accum = accum(Re_cnn)
	Re_rnn_accum = accum(Re_rnn)

	# do the plotting

	# first plot return vs. date
	plotter1 = Plotter('Weekly Return', dateLabel, 'Year', 'Return')
	plotter1.addLine(Re_bl, 'Baseline')
	plotter1.addLine(Re_oc, 'Oracle')
	plotter1.addLine(Re_cnn, 'CNN')
	plotter1.addLine(Re_rnn, 'RNN')
	plotter1.plot('results/Figures/Weekly_Re')

	'''
	plt.plot(x,Re_bl,'-')
	plt.plot(x,Re_oc,'-')
	plt.plot(x,Re_cnn,'-')
	plt.plot(x,Re_rnn,'-')
	plt.xticks(x, dateLabel)
	plt.title('Weekly Return')
	plt.legend(['Baseline','Oracle', 'CNN', 'RNN'])
	plt.xlabel('Year')
	plt.ylabel('Return')
	plt.savefig('results/Figures/Weekly_Re')
	plt.show()
	'''
	# then plot accumulated return vs. date
	plotter2 = Plotter('Weekly Return', dateLabel, 'Year', 'Accumulated Return')
	plotter2.addLine(Re_bl_accum, 'Baseline')
	plotter2.addLine(Re_oc_accum, 'Oracle')
	plotter2.addLine(Re_cnn_accum, 'CNN Test')
	plotter2.addLine(Re_rnn_accum, 'RNN Test')
	plotter2.plot('results/Figures/Weekly_Re_Accum')

	'''
	plt.plot(x,Re_bl_accum,'-')
	plt.plot(x,Re_oc_accum,'-')
	plt.plot(x,Re_cnn_accum,'-')
	plt.plot(x,Re_rnn_accum,'-')
	plt.xticks(x, dateLabel)
	plt.title('Weekly Accumulated Return')
	plt.legend(['Baseline','Oracle', 'CNN Test', 'RNN Test'])
	plt.xlabel('Year')
	plt.ylabel('Accumulated Return')
	plt.savefig('results/Figures/Weekly_Re_Accum')
	plt.show()
	'''



