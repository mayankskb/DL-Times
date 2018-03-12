import tensorflow as tf

a = tf.constant(6.5, name = 'constant_a')
b = tf.constant(3.4, name = 'constant_b')
c = tf.constant(3.0, name = 'constant_c')
d = tf.constant(100.2, name = 'constant_d')

square = tf.square(a, name = 'square_a')
power = tf.pow(b, c, name = 'pow_b_c')
sqrt = tf.sqrt(d, name = 'sqrt_d')

final_sum = tf.add_n([square, power, sqrt], name = 'add_n')

sess = tf.Session()

print('----------------------------------------------------')
print('Square of a : {}'.format(sess.run(square)))
print('Power of b ^ c : {}'.format(sess.run(power)))
print('Square root of d : {}'.format(sess.run(sqrt)))

print('Sum of square, power and square root : {}'.format(sess.run(final_sum)))
print('----------------------------------------------------')

writer = tf.summary.FileWriter('./tensormath_op', sess.graph)

writer.close()
sess.close()
