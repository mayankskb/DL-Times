import pandas as pd
import numpy as np
import statsmodels.api as sm

from return_data import read_goog_sp500_logistic_data

xData, yData = read_goog_sp500_logistic_data()

logit = sm.Logit(yData, xData)

# Fit the Logistic model
result = logit.fit()

# All values > 0.5 predict an up day for Google
prediction = (result.predict(xData) > 0.5)

# Count the number of times the actual up days match the predicted up days
num_accurate_predictions = (list(yData == prediction)).count(True)
pctAccuracy = float(num_accurate_predictions) / float(len(prediction))

print("Percentage Accuracy : {}".format(pctAccuracy))

###########################################################################
#
#  Simple Logistic regression with tf.contrib.learn
#
###########################################################################
import tensorflow as tf

features = [tf.contrib.layers.real_valued_column('x', dimension=1)]

estimator = tf.contrib.learn.LinearClassifier(feature_columns = features)

# All returns in a 2D array
# [[-0.02184618]
#  [ 0.00997998]
#  [ 0.04329069]
#  [ 0.03254923]
#  [-0.01781632]]
x = np.expand_dims(xData[:,0], axis = 1)

# True False value for up/down days in a 2D array
# [[False]
#  [True]
#  [True]
#  [True]]
y = np.expand_dims(np.array(yData), axis = 1)

input_fn = tf.contrib.learn.io.numpy_input_fn({'x' : x}, y, batch_size = 100,
                                                num_epochs = 100000)

fit = estimator.fit(input_fn = input_fn, steps = 100000)

results = fit.evaluate(input_fn = input_fn, steps = 1000)
print('Accuracy with Estimator: ', results)

for variable_name in fit.get_variable_names():
    print(variable_name, ' ---> ', fit.get_variable_value(variable_name))
