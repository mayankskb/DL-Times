import tensorflow as tf

x = tf.constant([100, 200, 300], name = 'x')
y = tf.constant([1, 2, 3], name = 'y')

sum_x = tf.reduce_sum(x, name = 'sum_x')
prod_y = tf.reduce_prod(y, name = 'prod_y')

final_div = tf.div(sum_x, prod_y, name = 'final_div')

final_mean = tf.reduce_mean([sum_x, prod_y], name = 'final_mean')

sess = tf.Session()

print('-------------------------------------------------------------------')
print('The value of x : {}'.format(sess.run(x)))
print('The value of y : {}'.format(sess.run(y)))
print('The value of sum_x : {}'.format(sess.run(sum_x)))
print('The value of prod_y : {}'.format(sess.run(prod_y)))
print('The value of final_div : {}'.format(sess.run(final_div)))
print('The value of final_mean : {}'.format(sess.run(final_mean)))
print('-------------------------------------------------------------------')

writer = tf.summary.FileWriter('./tensormath', sess.graph)

writer.close()
sess.close()
