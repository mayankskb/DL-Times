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
W = tf.Variable(tf.zeros(1,1))
b = tf.Variable(tf.zeros([1]))

# Placeholder to feed in the returns, returns have many rows,
# just one column
x = tf.placeholder(tf.float32, [None, 1])

Wx = tf.matmul(x, W)
y = Wx + b

# Placeholder to hold the y-labels, also returns
y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function
cost = tf.reduce_mean(tf.square(y_ - y))

# Initializing the optimizer
train_step_constant = tf.train.GradientDescentOptimizer(0.1).minimize(cost)