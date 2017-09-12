#
#   House_Price_Prediction.py
#
#   this is a very simple prediction of house price based on house size, implemented
#   in TensorFlow. This code is part of learning TensorFLow from Plural Sight
#   Practise code by Mayank Mishra

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#generate some house size between 1000 and 3500(typical sqft of house)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low = 1000, high = 3000, size = num_house)

#generate house prices depending upon the house size and some random figure
np.random.seed(42)
house_price = house_size * 100 + np.random.randint(low = 20000, high = 70000, size = num_house)

#plot the graph between house_size and house_price
plt.plot(house_size, house_price, "bx") #bx denotes the bluex
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

#need to normalise the data to prevent the underflow
def normalize(array):
    return (array - array.mean()) / array.std()

#1.Preparing data
#define number of training sample = 0.7 = 70%. We can take the first 70% as the data are generated randomly.
num_train_samples = math.floor(num_house * 0.7)

#training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_house_price = np.asanyarray(house_price[:num_train_samples])

train_house_size_norm = normalize(train_house_size)
train_house_price_norm = normalize(train_house_price)

#test data
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

#setup TensorFlow placeholder that get updated as we descend down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

#define the variable holding the size faactor and price we set during training.
#we initialize them to some random values based on the normal distribution.
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

#2.inference rule
#defining the operation for the predicting value. 
tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size), tf_price_offset)

#3.loss function - mean square error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price,2)) / (2 * num_train_samples)

#optimizer learning rate. The size of the steps down the gradient.
learning_rate = 0.1

#4.define a gradient descent optimizer that will minimize the loss defined in the operation "cost".
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#Initialize the variable 
init = tf.global_variables_initializer()

#launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    #set how often to display training progress and number of training iterations
    display_every = 2
    num_training_itr = 50
    
    #keep iteration in the training data
    for iteration in range(num_training_itr):

        #Fit all data
        for (x,y) in zip(train_house_size_norm,train_house_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price:y})

        #Display current status
        if(iteration + 1) % display_every == 0:
            c= sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_house_price_norm})
            print("iteration #: ","%04d" %(iteration + 1),"cost=", "{:.9f}".format(c), "size factor = ",sess.run(tf_size_factor)," price offset = ",sess.run(tf_price_offset))

    print("Optimization is Finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm, tf_price:train_house_price_norm})
    print("Trained cost = ",training_cost, "size factor = ", sess.run(tf_size_factor), "'price_offset :", sess.run(tf_price_offset))

    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_house_price.mean()
    train_price_std = train_house_price.std()

    #Plot the graph
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(train_house_size,train_house_price,"go",label = "Training data")
    plt.plot(test_house_size,test_house_price,"mo",label = "Testing data")
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
    (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
    label = "Learner=d Regression")
    plt.legend(loc = "upper left")
    plt.show()
