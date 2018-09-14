import tensorflow as tf


# y = Wx + B
x = tf.Variable([10, 100], tf.float32, name = 'x')

# Note that these tensors can hold tensor of data of any shape
W = tf.placeholder(tf.int32, name = 'W')
B = tf.placeholder(tf.int32, name = 'B')

y = W * x + B

# Initialize all variables defined
init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    print('Final result : Wx + B = {}'.format(sess.run(fetches = y, feed_dict = {W : [3, 33], B : [7, 9]})))

number = tf.Variable(2, name = 'number')
multiplier = tf.Variable(1, name = 'multiplier')

init = tf.global_variables_initializer()

result = number.assign(tf.multiply(number, multiplier))

with tf.Session() as sess:
    sess.run(init)

    for i in range(10):
        print('Result number * multiplier = ', sess.run(result))

        print('New value for number = ', sess.run(number))
        print('Increment multiplier, new value = ', sess.run(multiplier.assign_add(1)))
