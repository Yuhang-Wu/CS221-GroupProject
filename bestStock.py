def bestStock(SP):
	result_P = []
	result_R = []
	compNum = len(SP[0])

	# Note that the last day doesn't have a bestStock return since we don't know prices after that
	for timeStep in xrange(len(SP)-1):
		P = [0 for i in xrange(compNum)]
		bestReturn = -float('inf')
		bestIdx = 0
		for compIdx in xrange(compNum):
			sReturn = (SP[timeStep+1][compIdx] - SP[timeStep][compIdx]) / SP[timeStep][compIdx]
			if sReturn >= bestReturn:
				bestReturn = sReturn
				bestIdx = compIdx
		P[bestIdx] = 1
		result_P.append(P)
		result_R.append(bestReturn)
	return result_R

