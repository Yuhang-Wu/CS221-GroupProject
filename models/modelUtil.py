from __future__ import print_function
from utils import dataUtil as du
import tensorflow as tf
import numpy as np

Exp = np.exp(1.0)

def combineList(tensorList, weights):
	t = tensorList[0] * weights[0]
	for i in range(1, len(tensorList)):
		t += tensorList[i] * weights[i]
	return t

def getLastEleWeights(tensorList):
	out = [tf.constant(0.0, dtype = tf.float32) for _ in range(len(tensorList))]
	out[-1] = tf.constant(1.0, dtype = tf.float32)
	return out

def getAverageWeights(tensorList):
	l = len( tensorList )
	out = [ tf.constant( 1.0 / float(l), dtype = tf.float32 ) for _ in range(l) ]
	return out

def getDecayingWeights(tensorList):
	l = len(tensorList)
	cur = 1.0
	out = [cur]
	total = cur

	# create list
	for i in range(1, len(tensorList)):
		cur = cur * Exp
		out.append(cur)
		total += cur

	# normalize
	out = [ele/total for ele in out]

	# convert to tensor
	out = [tf.constant(ele, dtype = tf.float32) for ele in out]
	return out

def getAttentionWeights(tensorList):
	pass

# multiply a batch of matrix a tensor in the shape of [D, N, L]
# with a transformation matrix [L, transformSize]
# output a tensor of shape [D, N, transformSize]
def batchMatMul(a, b, D):
	out = []
	for i in range(D):
		out.append(tf.matmul(a[i], b))
	return tf.stack(out)

# assume the input is of shape [D, 1]
# add one more 0 to it (output shape [D+1, 1])
def addBias(x):
	x = tf.concat([x, tf.constant(np.array([[0.0]]), dtype = tf.float32)], axis = 0)
	return x

# get weight variavle of initial values from truncated normal
def tnVariable(shape, name = None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	if name is None:
		return tf.Variable(initial)
	else:
		return tf.get_variable(name = name, initializer = initial, dtype = tf.float32)

# get bias variable of all initial value of 0.1
def biasVariable(shape, name = None):
    initial = tf.constant(0.1, shape=shape)
    if name is None:
    	return tf.Variable(initial)
    else:
    	return tf.get_variable(name = name, initializer = initial, dtype = tf.float32)

# stride 1 convolution, H and W won't change
def conv2dStide1(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 1 by 1 average ???
def avg1x1(x):
	return tf.nn.avg_pool(x, ksize=[1, 1, 1, 1],
                          strides=[1, 1, 1, 1], padding='SAME')

def calcProfit(action, logR, inBatch = False):
	if not inBatch:
		return calcProfitNoBatch(action, logR)
	else:
		return calcProfitBatch(action, logR)

def calcTransCost(action, prevAction, prevLogR, transCostParams, mRatio, inBatch = False):
	if not inBatch:
		return calcTransCostNoBatch(action, prevAction, prevLogR, transCostParams, mRatio)
	else:
		return calcTransCostBatch(action, prevAction, prevLogR, transCostParams, mRatio)

def calcProfitNoBatch(action, logR):
	profit = tf.reduce_sum(tf.multiply(action[:-1], logR))
	return profit

def calcProfitBatch(action, logR):
	profit = tf.reduce_sum(tf.multiply(action[:,:-1], logR), axis = 1)
	return tf.reduce_sum(profit)


def calcTransCostNoBatch(action, prevAction, prevLogR, transCostParams, mRatio):
	c = transCostParams['c']
	c0 = transCostParams['c0']
	priceRatio = tf.exp(prevLogR)
	changes = tf.abs(action[:-1] - mRatio * tf.multiply(priceRatio, prevAction[:-1]))
	transactionCost = tf.reduce_sum( tf.multiply(c, changes) )
	transactionCost += c0
	return transactionCost

def calcTransCostBatch(action, prevAction, prevLogR, transCostParams, mRatio):
	c = transCostParams['c']
	c0 = transCostParams['c0']
	priceRatio = tf.exp(prevLogR)
	changes = tf.abs(action[:,:-1] - mRatio * tf.multiply(priceRatio, prevAction[:,:-1]))
	transactionCost = tf.reduce_sum( tf.multiply(c, changes) , axis = 1)
	transactionCost += c0
	return tf.reduce_sum(transactionCost)

def train1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess, B = None):
	return trainOrTest1Epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess)

def test1epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess, B = None):
	return trainOrTest1Epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess, training = False)

def trainOrTest1Epoch(returnTensor, prevReturnMatrix, nextReturnMatrix, curModel, sess, training = True):	
	totalIters = returnTensor.shape[0]
	prevLoss = 0.0
	D = len(prevReturnMatrix[0])
	prevA = du.getInitialAllocation(D)
	allActions = []
	allLosses = []

	for t in range(totalIters):
		mRatio = du.loss2mRatio(prevLoss)
		inputs = {
			'X': returnTensor[t],
			'prevReturn': prevReturnMatrix[t],
			'nextReturn': nextReturnMatrix[t],
			'prevA': prevA,
			'mRatio': mRatio
		}
		if training:
			curA, curLoss = curModel.train(inputs, sess)
		else:
			curA, curLoss = curModel.get_action(inputs, sess)
		allActions.append(curA)
		allLosses.append(curLoss)

		prevLoss = curLoss
		prevA = curA
	
	totalLoss = sum(allLosses)
	growthRates = map(lambda x: 1-x, allLosses)

	return allActions, growthRates

	# add one more dimension for batching (which seems not very useful in this case)	
def addNoneDim(shape):
	return tuple([None] + list(shape))

def addNoneDimToAll(shapes):
	return [addNoneDim(shape) for shape in shapes]

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



