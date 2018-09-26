# coding=utf-8
import os
import numpy as np
import tensorflow as tf
import input_data
import MainModel

from PIL import Image
import matplotlib.pyplot as plt

N_CLASSES = 2  # Number of calss 0，1
IMG_W = 208  # width of img
IMG_H = 208  # height of img
BATCH_SIZE = 16  # batch size
CAPACITY = 2000  # queue capacity 2000
MAX_STEP = 10000  # Max step for gradient decent
learning_rate = 0.0001  # learning rate

"""
 training function
"""

def run_training():
    """
    #1.img processing
    """
    # the path of the img
    train_dir = '/Users/gao han/Downloads/images/train/'
    # path to save the log
    logs_train_dir = '/Users/gao han/Downloads/images/log/'

    # path to save parameters of the model
    train_model_dir = '/Users/gao han/Downloads/images/model/'

    # get the img and the label
    train, train_label = input_data.get_files(train_dir)

    #  get batch TensorFlow
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)

    """
    ##2. CNN
    """
    # get the output
    train_logits = MainModel.inference(train_batch, BATCH_SIZE, N_CLASSES)

    """
    ##3. crossover entropy and gradient descent optimizer 
    """
    train_loss = MainModel.losses(train_logits, train_label_batch)

    train_op = MainModel.trainning(train_loss, learning_rate)

    """
    ##4.Define the variables 
    """
    # Calculate the classification accuracy
    train__acc = MainModel.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()

    sess = tf.Session()

    # save the log to logs_train_dir
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    # Initializing Variables
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            # print the loss and accuracy each 50 steps
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))

                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            # The model is saved every 2,000 steps
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


    # This exception is thrown if read to the end of the file queue
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()  # coord.request_stop() to issue a command to terminate all threads

    coord.join(threads)
    sess.close()


def get_one_image(data):

    n = len(data)
    ind = np.random.randint(0, n)
    img_dir = data[ind]

    image = Image.open(img_dir)
    plt.legend()
    plt.imshow(image)   # show img
    image = image.resize([208, 208])
    image = np.array(image)
    return image


def get_one_image_file(img_dir):
    image = Image.open(img_dir)
    plt.legend()
    plt.imshow(image)  # show img
    image = image.resize([208, 208])
    image = np.array(image)
    return image


"""
evaluate one image
"""


def evaluate_one_image():

    image_array = get_one_image_file("/Users/gao han/Downloads/images/test1/52.jpg")

    with tf.Graph().as_default():
        BATCH_SIZE = 1  # get one img
        N_CLASSES = 2  # two classification

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])  # image resize
        logit = MainModel.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)  # Add activation function

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # model save path
        logs_train_dir = '/Users/gao han/Downloads/images/model/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            # download the model paras
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)

            if ckpt and ckpt.model_checkpoint_path:

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)

                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            # get max prediction
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a cat with possibility %.6f' % prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' % prediction[:, 1])



"""
main function
"""


def main():
    #run_training() #training the model and get the paras
    evaluate_one_image()


if __name__ == '__main__':
    main()