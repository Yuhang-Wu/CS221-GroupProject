from __future__ import print_function
import tensorflow as tf

def calcReturn(prevA, prevS, nextS, mRatio, action, transCostParams, D, N):
	# return the total return

	# get parameters
	c = transCostParams['c']
	c0 = transCostParams['c0']

	# get holdings (stock holdings except for reserve) and stock prices
	todayHoldings = tf.slice(action, [0, 0], [D, 1])
	yesterdayHoldings = tf.slice(prevA, [0, 0], [D, 1])
	todayS = tf.slice(prevS, [0, N-1], [D, 1])
	yesterdayS = tf.slice(prevS, [0, N-2], [D, 1])

	# calc profit and transaction cost
	profit = calcProfit(nextS, todayS, todayHoldings)
	transactionCost = calctransactionCost(todayS, yesterdayS, todayHoldings, yesterdayHoldings, mRatio, c, c0)

	# total return
	R = profit - transactionCost
	print('R', R.shape)
	return R

	# profit
def calcProfit(nextS, todayS, todayHoldings):
	stockChange = nextS - todayS
	profit = tf.reduce_sum( tf.multiply( tf.div(stockChange, todayS), todayHoldings) )
	return profit

	# transaction cost
def calctransactionCost(todayS, yesterdayS, todayHoldings, yesterdayHoldings, mRatio, c, c0):
	holdingsChange = tf.abs( tf.div(todayHoldings, todayS) - mRatio * tf.div(yesterdayHoldings, yesterdayS) )
	transactionCostEach = tf.multiply(c, tf.multiply(holdingsChange, todayS))
	transactionCostTotal = tf.reduce_sum(transactionCostEach) + c0
	return transactionCostTotal
