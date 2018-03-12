import tensorflow as tf


# y = Wx + B
x = tf.constant([10, 100], name = 'x')

# Note that these tensors can hold tensor of data of any shape
W = tf.placeholder(tf.int32, name = 'W')
B = tf.placeholder(tf.int32, name = 'B')

Wx = tf.multiply(W, x, name = 'Wx')

y = tf.add(Wx, B, name = 'y')

with tf.Session() as sess:
    print('Intermediate result : Wx = {}'.format(sess.run(fetches = Wx, feed_dict = {W : [3, 33]})))
    print('Final result : Wx + B = {}'.format(sess.run(fetches = y, feed_dict = {W : [3, 33], B : [7, 9]})))


writer = tf.summary.FileWriter('./fetches_feed_dict', sess.graph)
writer.close()
