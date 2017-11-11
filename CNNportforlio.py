# more to be clarified
# log-return as input
# softmax probability as output
# reward function: return for the future period
# Question! how to integrate the previous portfolio?


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def avg_pool_4x1(x):
    return tf.nn.avg_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')

# first convolution layer
W_conv1 = weight_variable([1, 4, 1, 10])
b_conv1 = bias_variable([10])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = avg_pool_4x1(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([1, 4, 10, 20])
b_conv2 = bias_variable([20])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avg_pool_4x1(h_conv2)

# densely connected layer
W_fc1 = weight_variable([__, 2 ** K])
b_fc1 = bias_variable([2 ** K])
h_pool2_flat = tf.reshape(h_pool2, [-1, _*_*20])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# readout layer
W_fc2 = weight_variable([2 ** K, K])
b_fc2 = bias_variable([K])

# train and evaluate the model
# need modified here!!!!!!!
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
            
            print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

