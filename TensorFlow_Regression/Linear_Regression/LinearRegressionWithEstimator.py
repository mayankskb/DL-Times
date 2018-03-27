import numpy as np
from sklearn import datasets, linear_model

from return_data import read_goog_sp500_data

xData, yData = read_goog_sp500_data()

# Set up a linear model to represent this
googModel = linear_model.LinearRegression()

googModel.fit(xData.reshape(-1, 1), yData.reshape(-1, 1))

print('Coefficient : ', googModel.coef_)
print('Intercept : ', googModel.intercept_)



############################################################
#
#  Simple Linear Regression with tf.contrib.learn
#
############################################################
import tensorflow as tf

features = [tf.contrib.layers.real_valued_column('x', dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns = features)

input_fn = tf.contrib.learn.io.numpy_input_fn({'x' : xData}, yData, batch_size = len(xData),
                                                num_epochs = 100000)

fit = estimator.fit(input_fn = input_fn, steps = 100000)

for variable_name in fit.get_variable_names():
    print(variable_name, ' ---> ', fit.get_variable_value(variable_name))
