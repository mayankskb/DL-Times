import tensorflow as tf

a = tf.constant(6, name = 'constant_a')
b = tf.constant(3, name = 'constant_b')
c = tf.constant(10, name = 'constant_c')
d = tf.constant(5, name = 'constant_d')

mul = tf.multiply(a, b, name = 'mul')
div = tf.div(c, d, name = 'div')


addn = tf.add_n([mul, div], name = 'add_n')

print(addn)
print(a)

sess = tf.Session()

print(sess.run(addn))
sess.run(addn)      # will not print data becoz only session object is able to get value of the tensors


# visualisation it usintg tensorboard
writer = tf.summary.FileWriter('./computationgraph_logs', sess.graph)
# first argument is directory where events to store
# second argument is what to write
writer.close()
sess.close()
