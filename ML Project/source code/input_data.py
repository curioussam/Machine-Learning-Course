import tensorflow as tf
import numpy as np
import os

""" 
 # the path of the img
"""
train_dir = "/Users/gao han/Downloads/images/train"

""" 
 processing data
"""


def get_files(file_dir):
    cats = []
    lable_cats = []
    dogs = []
    lable_dogs = []

    # processing all the img in the path
    for file in os.listdir(file_dir):
        name = file.split('.')  # split name
        # name format [‘dog’，‘9981’，‘jpg’]
        if name[0] == 'cat':
            cats.append(file_dir + "/" + file)
            lable_cats.append(0)  # set cat label as 0 dog as 1
        else:
            dogs.append(file_dir + "/" + file)
            lable_dogs.append(1)
    print(" %d cat, %d dog" % (len(cats), len(dogs)))
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((lable_cats, lable_dogs))

    # merge the two list
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp) # disorder the list

    image_list = list(temp[:, 0])  # img is the first column
    label_list = list(temp[:, 1])  # label is the second column
    label_list = [int(i) for i in label_list]

    return image_list, label_list


""" 
transfer the image to tensorFlow 
"""


def get_batch(image, label, image_W, image_H, batch_size, capacity):

    image = tf.cast(image, tf.string)  # image to string
    label = tf.cast(label, tf.int32)  # label to int
    # put into the queue
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    # decode image to tensor
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # use image_W,image_H to resize the image
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # standardization image
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)  # image to float32

    return image_batch, label_batch