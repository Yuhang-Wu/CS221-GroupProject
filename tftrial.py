from __future__ import print_function
import tensorflow as tf
import numpy as np

'''
A baby tensorflow example
Note print needs parenthesis
'''
def main():
	testTFinitializer()
	pass

def testTFinitializer():
	print(tf.contrib.layers.xavier_initializer())
def testTFindex():
	a = np.array(np.random.rand(2,3))
	tfa = tf.constant(a, dtype = tf.float32)
	print(tfa)
	tfb = tfa[:,:-1]
	print(tfb)


def tftrial():
	print(tf.__version__)

	# declare constant
	b = tf.constant(3.0, dtype = tf.float32)
	print(b)

	# declare variable (parameters of the model)
	W = tf.Variable([[0.0, 0.0, 0.0]], dtype = tf.float32, name = 'W')
	print(W)

	# declare placeholder (what are the inputs)
	x = tf.placeholder(dtype = tf.float32, shape = (3,1))
	y = tf.placeholder(dtype = tf.float32)
	print(x)
	print(y)

	# build the model -> a computational graph that describes how out model works
	model = tf.matmul(W, x) + b

	# define the loss
	loss = tf.reduce_sum(tf.square(model - y))

	# initialize variables (necessary!)
	init_op = tf.global_variables_initializer()

	# declare how to train our model
	step_size = 0.1
	optimizer = tf.train.GradientDescentOptimizer(step_size)
	train = optimizer.minimize(loss)

	# run the session
	with tf.Session() as sess:

		# initialize variables
		sess.run(init_op)

		# what our data would be
		feed_dict = { 
						x: np.array([[1,2,1]]).T , 
					 	y: 1.0
					}

		# training 
		for _ in range(10):
			sess.run(train, feed_dict = feed_dict)

			# print out weight and loss
			print('W:', sess.run(W, feed_dict = feed_dict))
			print('loss:', sess.run(loss, feed_dict = feed_dict))

if __name__ == '__main__':
	main()
