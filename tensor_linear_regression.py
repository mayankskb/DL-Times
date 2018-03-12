import tensorflow as tf

# Model Parameter
W = tf.Variable([.3], dtype = tf.float32)
B = tf.Variable([-.3], dtype = tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + B

y = tf.placeholder(tf.float32)

#loss function
loss = tf.reduce_sum(tf.square(linear_model - y))

# Optimizer function
optimizer = tf.train.GradientDescentOptimizer(0.01)    #learning rate 0.01

train = optimizer.minimize(loss)

# Setting up the training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# initialize the global variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train, {x : x_train, y : y_train})

    # evaluate training accuracy
    curr_W, curr_B, curr_loss = sess.run([W, B, loss], {x : x_train, y : y_train})

    print('Current Slope : {}'.format(curr_W))
    print('Current Intercept : {}'.format(curr_B))
    print('Current Loss : {}'.format(curr_loss))
