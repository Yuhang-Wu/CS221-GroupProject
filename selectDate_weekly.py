
# Select the dates which are the first date of each week in database
# store the result as a list of (date, index) tuple
def selectDate_weekly(allfilecontents):
	cws = allfilecontents[0][1][1][0] #the first day of current week 
	cws = cws[0:4] + cws[5:7] + cws[8:10]
	cws_int = int(cws) #covert string to integer
	dateSelected = []
	for dateIdx in xrange(len(allfilecontents[0][1][1:])):
		oneDay = allfilecontents[0][1][1:][dateIdx][0]
		oneDay_int = int(oneDay[0:4] + oneDay[5:7] + oneDay[8:10])
		if oneDay_int >= cws_int:
			dateSelected.append((oneDay, dateIdx))
			# update cws and cws_int
			if cws[4:6] =='12': #may need to drump to the next year
				if int(cws[6:8]) >= 25:
					cws = str(int(cws[0:4]) +1).zfill(4) + str('01') + str(int(cws[6:8])+7-31).zfill(2)
					cws_int = int(cws)
				else:
					cws_int += 7
					cws = str(cws_int)
			elif cws[4:6] in ['01', '03', '05', '07', '08', '10']:
				if int(cws[6:8]) >= 25:
					cws = str(int(cws[0:4])).zfill(4) + str(int(cws[4:6])+1).zfill(2) + str(int(cws[6:8])+7-31).zfill(2)
					cws_int = int(cws)
				else:
					cws_int += 7
					cws = str(cws_int)
			elif cws[4:6] == '02':
				if int(cws[0:4]) % 4 ==0: #run nian
					if int(cws[6:8]) >= 23:
						cws = str(int(cws[0:4])).zfill(4) + str('03') + str(int(cws[6:8])+7-29).zfill(2)
						cws_int = int(cws)
					else:
						cws_int += 7
						cws = str(cws_int)
				else: #pin nian
					if int(cws[6:8]) >= 22:
						cws = str(int(cws[0:4])).zfill(4) + str('03') + str(int(cws[6:8])+7-28).zfill(2)
						cws_int = int(cws)
					else:
						cws_int += 7
						cws = str(cws_int)
			else:
				if int(cws[6:8]) >= 24:
					cws = str(int(cws[0:4])).zfill(4) + str(int(cws[4:6])+1).zfill(2) + str(int(cws[6:8])+7-30).zfill(2)
					cws_int = int(cws)
				else:
					cws_int += 7
					cws = str(cws_int)
	return dateSelected[1:]
