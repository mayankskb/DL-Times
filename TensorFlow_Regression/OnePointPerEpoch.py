import numpy as np
from sklearn import datasets, linear_model

from return_data import read_goog_sp500_data

xData, yData = read_goog_sp500_data()

# Set up a linear model to represent this
googModel = linear_model.LinearRegression()

googModel.fit(xData.reshape(-1, 1), yData.reshape(-1, 1))

print(googModel.coef_)
print(googModel.intercept_)



############################################################
#
#  Simple Regression - one point per epoch
#
############################################################

import tensorflow as tf

# Model linear regression y = Wx + b
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

# Placeholder to feed in the returns, returns have many rows,
# just one column
x = tf.placeholder(tf.float32, [None, 1])

Wx = tf.matmul(x, W)
y = Wx + b

# Add summary ops to collect data
W_hist = tf.summary.histogram('weights', W)
b_hist = tf.summary.histogram('biases', b)
y_hist = tf.summary.histogram('y', y)

# Placeholder to hold the y-labels, also returns
y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function
cost = tf.reduce_mean(tf.square(y_ - y))

cost_hist = tf.summary.histogram('cost', cost)

# Initializing the optimizer
train_step_constant = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


# Set up a method to perform the actual training. Allow us to
# modify the optimizer used and also the number of steps
# in the training
def trainWithOnePointPerEpoch(steps, train_step):

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		merged_summary = tf.summary.merge_all()
		writer = tf.summary.FileWriter('./OnePointPerEpoch_logs', sess.graph)

		for i in range(steps):

			# Extract one training point
			xs = np.array([[xData[i % len(xData)]]])
			ys = np.array([[yData[i % len(yData)]]])

			feed = { x: xs, y_: ys}

			sess.run(train_step, feed_dict = feed)

			# Write out histogram summaries
			result = sess.run(merged_summary, feed_dict = feed)
			writer.add_summary(result, i)

			# Print result to screen for every 1000 iteration
			if(i + 1) % 1000 == 0:
				print('After %d iteration:' % i)

				print('W: %f' % sess.run(W))
				print('b: %f' % sess.run(b))

				print('cost: %f' % sess.run(cost, feed_dict = feed))

			writer.close()


trainWithOnePointPerEpoch(10000, train_step_constant)