import tensorflow as tf
import matplotlib.image as mp_img
import matplotlib.pyplot as plt
import os

filename = 'dandelion-sky-flower-nature.jpeg'
image = mp_img.imread(filename)

print("Image Shape : ", image.shape)
print("Image Array : ", image)

plt.imshow(image)
plt.show()

x = tf.Variable(image, name = 'x')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

#    transpose = tf.transpose(x, perm = [1, 0, 2])
    transpose = tf.image.transpose_image(x)

    result = sess.run(transpose)
    print("Image Shape : ", result.shape)
    print("Image Array : ", result)

    plt.imshow(result)
    plt.show()
