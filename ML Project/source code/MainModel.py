import tensorflow as tf

"""
##1. build networks
"""


def inference(images, batch_size, n_classess):
    """
    The first convolution layer
    """

    with tf.variable_scope('conv1') as scope:
        # Initialization weight，[3,3,3,16]
        weights = tf.get_variable('weights', shape=[3, 3, 3, 16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        # Initialization bias
        biases = tf.get_variable('biases', shape=[16], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the bias to the result
        pre_activation = tf.nn.bias_add(conv, biases)
        # relu activation function
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    """
    Pooling layer
    """
    with tf.variable_scope('pooling1_lrn') as scope:
        # tf.nn.max_pool:The forward propagation of maximum pooling layer
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='poolong1')
        # Local Response Normalization: A method for improving accuracy after activation and pooling.
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    """
    Second convolution layer
    """
    #  same as the first layer
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 16, 16], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    """
    Second Pooling layer
    """
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1, alpha=0.001 / 9, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')

    """
     local3 fully connected layer 
    """
    with tf.variable_scope('local3') as scope:

        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        # get the number of columns of reshape
        # matrix dot product to meet the number of columns equal to the number of rows
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    """
     local4 fully connected layer 
    """
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', shape=[128, 128], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[128], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
    """
     soft max liner logistics
    """
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear', shape=[128, n_classess], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[n_classess], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


"""
##2.define loss function(Cross entropy loss function)
"""


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        # logits Represents the output of the neural network，labels means right answer
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        # get cross_entropy mean
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


"""
##3.Using gradient descent to minimize the loss function
"""


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


"""
##4.Define evaluation function
"""


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        # Whether the output are equal to the label
        correct = tf.nn.in_top_k(logits, labels, 1)

        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)  # get accuracy

        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

