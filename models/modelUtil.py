from __future__ import print_function
import tensorflow as tf
import numpy as np

def calcProfit(action, logR):
	profit = tf.reduce_sum(tf.multiply(action[:-1], logR))
	return profit

def calcTransCost(action, prevAction, prevLogR, transCostParams, mRatio):
	# get parameters
	c = transCostParams['c']
	c0 = transCostParams['c0']
	priceRatio = tf.exp(prevLogR)
	changes = tf.abs(action[:-1] - mRatio * tf.multiply(priceRatio, prevAction[:-1]))
	transactionCost = tf.reduce_sum( tf.multiply(c, changes) )
	transactionCost += c0
	return transactionCost


###
###
### below are deprecated functions please don't use
def calcReturnWithPrice(prevA, prevS, nextS, mRatio, action, transCostParams, D, N):
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
	profit = calcProfitWithPrice(nextS, todayS, todayHoldings)
	transactionCost = calctransactionCostWithPrice(todayS, yesterdayS, todayHoldings, yesterdayHoldings, mRatio, c, c0)

	# total return
	R = profit - transactionCost
	print('R', R.shape)
	return R

	# profit
def calcProfitWithPrice(nextS, todayS, todayHoldings):
	stockChange = nextS - todayS
	profit = tf.reduce_sum( tf.multiply( tf.div(stockChange, todayS), todayHoldings) )
	return profit

	# transaction cost
def calctransactionCostWithPrice(todayS, yesterdayS, todayHoldings, yesterdayHoldings, mRatio, c, c0):
	holdingsChange = tf.abs( tf.div(todayHoldings, todayS) - mRatio * tf.div(yesterdayHoldings, yesterdayS) )
	holdingsChange = tf.multiply(holdingsChange, todayS)
	transactionCostEach = tf.multiply(c, tf.multiply(holdingsChange, todayS))
	transactionCostTotal = tf.reduce_sum(transactionCostEach) + c0
	return transactionCostTotal



