
import EigenPortfolio as ep
import oracle as oa
import matplotlib.pyplot as plt
import CNNportforlio as cnn
import math

Re, D, M= ep.main()

# get return from senarios other than baseline
Re2 = oa.main()
Re3 = cnn.main()

# adjust off-set:
# Baseline omits the last day in both cnn and oracle
Re2 = Re2[:-1]
Re3 = Re3[:-1]
# Re3 should be shortest
Re = Re[len(Re)-len(Re3):]
Re2 = Re2[len(Re2)-len(Re3):]


# Find the right place to put label '2016' and '2017'
idx_2016 = -1
idx_2017 = -1
for idx, date in enumerate(D):
	if date[2:4] == '16':
		if idx_2016 == -1:
			idx_2016 = idx
	if date[2:4] == '17':
		if idx_2017 == -1:
			idx_2017 = idx
			break

Date = ['' for i in xrange(len(Re3))]
Date[idx_2016] = '2016'
Date[idx_2017] = '2017'
x = xrange(len(Re3))


'''
print len(Re)
print len(Re2)
print len(Re3)
'''




# get the (log) accumulated return for each senario
def accum(Re):
	result = []
	result.append(Re[0])
	for idx in xrange(1, len(Re)):
		cur = result[-1] + Re[idx]
		result.append(cur)
	return result

Re_accum = accum(Re)
Re2_accum = accum(Re2)
Re3_accum = accum(Re3)



# do the ploting

plt.plot(x,Re,'*-')
plt.plot(x,Re2,'*-')
plt.plot(x,Re3,'*-')
plt.xticks(x, Date)
plt.title('Weekly Return')
plt.legend(['Baseline','Oracle', 'CNN'])
plt.xlabel('Date')
plt.ylabel('Return')
plt.savefig('Weekly_Re')
#plt.margins(0.2)
#plt.subplots_adjust(bottom=0.15)
plt.show()

plt.plot(x,Re_accum,'*-')
plt.plot(x,Re2_accum,'*-')
plt.plot(x,Re3_accum,'*-')
plt.xticks(x, Date)
plt.title('Weekly Accumulated Return')
plt.legend(['Baseline','Oracle', 'CNN'])
plt.xlabel('Date')
plt.ylabel('Accumulate Return')
plt.savefig('Weekly_Re_Accum')
#plt.margins(0.2)
#plt.subplots_adjust(bottom=0.15)
plt.show()

