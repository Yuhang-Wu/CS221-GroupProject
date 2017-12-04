def bestStock(SP):
	result_P = []
	result_R = []
	compNum = len(SP[0])

	# Note that the last day doesn't have a bestStock return since we don't know prices after that
	for i in xrange(len(SP)-1):
		P = [0 for i in xrange(compNum)]
		bestReturn = -float('inf')
		bestIdx = 0
		for compIdx in xrange(compNum):
			sReturn = (SP[i+1][compIdx] - SP[i][compIdx]) / SP[i][compIdx]
			if sReturn >= bestReturn:
				bestReturn = sReturn
				bestIdx = compIdx
		P[bestIdx] = 1
		result_P.append(P)
		result_R.append(bestReturn)
	#print result_P
	return result_R
