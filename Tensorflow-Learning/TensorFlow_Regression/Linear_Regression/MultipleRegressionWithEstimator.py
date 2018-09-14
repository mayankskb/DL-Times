import numpy as np
from sklearn import datasets, linear_model

from return_data import read_xom_oil_nasdaq_data

nasdaqData, oilData, xomData = read_xom_oil_nasdaq_data()

combined = np.vstack((nasdaqData, oilData)).T

xomNasdaqOilModel = linear_model.LinearRegression()

xomNasdaqOilModel.fit(combined, xomData)
xomNasdaqOilModel.score(combined, xomData)

print('Coefficient : ', xomNasdaqOilModel.coef_)
print('Intercept : ', xomNasdaqOilModel.intercept_)


##########################################################################
#
#    Simple Multiple Regression with tf.contrib.learn
#
##########################################################################

import tensorflow as tf

features = [tf.contrib.layers.real_valued_column('nasdaq_x', dimension=1),
            tf.contrib.layers.real_valued_column('oil_x', dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns = features)

input_fn = tf.contrib.learn.io.numpy_input_fn({'nasdaq_x' : nasdaqData, 'oil_x' : oilData},
                                                y = xomData,
                                                batch_size = len(nasdaqData),
                                                num_epochs = 100000)

fit = estimator.fit(input_fn = input_fn, steps = 100000)

for variable_name in fit.get_variable_names():
    print(variable_name, ' ---> ', fit.get_variable_value(variable_name))
