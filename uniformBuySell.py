def ubs(SP):
	result_R = []
	compNum = len(SP[0])
	P = [1.0/compNum for i in xrange(compNum)]

	# Note that the last day doesn't have a bestStock return since we don't know prices after that
	for i in xrange(len(SP)-1):
		totalReturn = 0
		for compIdx in xrange(compNum):
			sReturn = (SP[i+1][compIdx] - SP[i][compIdx]) / SP[i][compIdx]
			totalReturn += sReturn * P[compIdx]
		result_R.append(totalReturn)
	return result_R