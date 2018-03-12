import tensorflow as tf
from PIL import Image

original_image_list = ['images/black_woods.jpg',
                        'images/cute-white-puppy.jpg',
                        'images/indian_actor.jpg',
                        'images/old_ruins.jpeg']


# Make a queue of file names including all the images specified--- Queues construct in tensor flow to read the image files
filename_queue = tf.train.string_input_producer(original_image_list)


# Read an entire image file
image_reader = tf.WholeFileReader()

# session object in tensor flow is a multi threaded session
with tf.Session() as sess:
    # Coordinate the loading of image files. This makes working with threads and queues very very easy
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    image_list = []
    for i in range(len(original_image_list)):
        # Read a whole file from the queue, the first returned value in the tuple is the
        # filename which we are ignoring.
        _, image_file = image_reader.read(filename_queue)

        # Decode the image file as a JPEG file, this will turn it into a Tensor which we can
        # then use in training
        image = tf.image.decode_jpeg(image_file)

        # Get a tensor of resize image
        image = tf.image.resize_images(image, [300, 300])
        image.set_shape((300,300,3))

        image = tf.image.flip_up_down(image)
        image = tf.image.central_crop(image, central_fraction = 0.5)
        #Get an image tenso and print its value
        image_array = sess.run(image)
        print(image_array.shape)

        # Converts a numpy array of the kind (300, 300, 3) to a Tensor of shape(300,300,3)
        image_tensor = tf.stack(image_array)

        print(image_tensor)

        # The expand_dims method add a new dimension
        image_list.append(image_tensor)

    # Finishes off the filename queue Coordinator
    coord.request_stop()
    coord.join(threads)

    # Converts all tensor to a single tensor with a 4th dimension
    # 4 images of (300, 300, 3) can be accessed as (0, 300, 300, 3),
    # (1, 300, 300, 3), (2, 300, 300, 3), etc.
    image_tensor = tf.stack(image_list)
    print(image_tensor)

    summary_writer = tf.summary.FileWriter('./image_transformation', graph = sess.graph)

    # Write out all the images in one go
    summary_str = sess.run(tf.summary.image('image', image_tensor, max_outputs = 4))
    summary_writer.add_summary(summary_str)

    summary_writer.close()
