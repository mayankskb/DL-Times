import numpy as np
from sklearn import datasets, linear_model

from return_data import read_xom_oil_nasdaq_data

nasdaqData, oilData, xomData = read_xom_oil_nasdaq_data()

combined = np.vstack((nasdaqData, oilData)).T

xomNasdaqOilModel = linear_model.LinearRegression()

xomNasdaqOilModel.fit(combined, xomData)
xomNasdaqOilModel.score(combined, xomData)

print(xomNasdaqOilModel.coef_)
print(xomNasdaqOilModel.intercept_)


##########################################################################
#
#    Multiple Regression
#
##########################################################################

import tensorflow as tf

# Model linear regression y = W1x1 + W2x2 + b
nasdaq_W = tf. Variable(tf.zeros([1, 1]), name = 'nasdaq_W')
oil_W = tf. Variable(tf.zeros([1,1]), name = 'oil_W')

b = tf.Variable(tf.zeros([1]), name = 'b')

nasdaq_x = tf.placeholder(tf.float32, [None, 1], name = 'nasdaq_x')
oil_x = tf.placeholder(tf.float32, [None, 1], name = 'oil_x')

nasdaq_Wx = tf.matmul(nasdaq_x, nasdaq_W)
oil_Wx = tf.matmul(oil_x, oil_W)

y = nasdaq_Wx + oil_Wx + b

y_ = tf.placeholder(tf.float32, [None, 1])

cost = tf.reduce_sum(tf.square(y_ - y))

train_step_ftrl = tf.train.FtrlOptimizer(1).minimize(cost)

all_x_nasdaq = nasdaqData.reshape(-1, 1)
all_x_oil = oilData.reshape(-1, 1)
all_ys = xomData.reshape(-1,1)

dataset_size = len(oilData)

def trainWithMultiplePointsPerEpoch(steps, train_step, batch_size):
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)

		
		for i in range(steps):

			if dataset_size == batch_size:
				batch_start_idx = 0
			elif dataset_size < batch_size:
				raise ValueError('dataset_size : %d, must be greater than batch_size : %d' % (dataset_size, batch_size))
			else:
				batch_start_idx = (i * batch_size) % dataset_size

			batch_end_idx = batch_start_idx + batch_size

			# Access the x and y values in the batches
			batch_x_nasdaq = all_x_nasdaq[batch_start_idx : batch_end_idx]
			batch_x_oil = all_x_oil[batch_start_idx : batch_end_idx]
			batch_ys = all_ys[batch_start_idx : batch_end_idx]

			# Reshape the 1-D arrays as 2-D festure vectors with many rows and 1 column
			feed = { nasdaq_x: batch_x_nasdaq, oil_x: batch_x_oil, y_: batch_ys }

			sess.run(train_step, feed_dict=feed)

			# Print result to screen for every 500 iterations
			if (i + 1) % 500 == 0:
				print('After %d iteration : ' % i)
				print('W1 : %f' % sess.run(nasdaq_W))
				print('W2 : %f' % sess.run(oil_W))
				print('b : %f' % sess.run(b))

				print('cost : %f' % sess.run(cost, feed_dict = feed))


trainWithMultiplePointsPerEpoch(10000, train_step_ftrl, len(oilData))