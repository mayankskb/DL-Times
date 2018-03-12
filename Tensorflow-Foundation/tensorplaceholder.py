import tensorflow as tf

x = tf.placeholder(tf.int32, shape = [3], name = 'x')
y = tf.placeholder(tf.int32, shape = [3], name = 'y')

sum_x = tf.reduce_sum(x, name = 'sum_x')
prod_y = tf.reduce_prod(y, name = 'prod_y')

final_div = tf.div(sum_x, prod_y, name = 'final_div')

final_mean = tf.reduce_mean([sum_x, prod_y], name = 'final_mean')

sess = tf.Session()

print('-------------------------------------------------------------------')
print('The value of x : {}'.format(sess.run(x, feed_dict = {x:[100, 200, 300]})))
print('The value of y : {}'.format(sess.run(y, feed_dict = {y:[1, 2, 3]})))
print('The value of sum_x : {}'.format(sess.run(sum_x, feed_dict = {x:[100, 200, 300]})))
print('The value of prod_y : {}'.format(sess.run(prod_y, feed_dict = {y:[1, 2, 3]})))
print('The value of final_div : {}'.format(sess.run(final_div, feed_dict = {x:[100, 200, 300], y:[1, 2, 3]})))
print('The value of final_mean : {}'.format(sess.run(final_mean, feed_dict = {x:[100, 200, 300], y:[1, 2, 3]})))
print('-------------------------------------------------------------------')

writer = tf.summary.FileWriter('./tensorplaceholder', sess.graph)

writer.close()
sess.close()
