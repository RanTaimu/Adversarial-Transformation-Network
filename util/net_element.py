import tensorflow as tf


def weight_variable(shape, name):
    """
    """
    initializer = tf.truncated_normal(shape, stddev=0.01)
    return tf.get_variable(initializer=initializer, name=name)


def bias_variable(shape, name):
    """
    """
    initializer = tf.constant(0.1, shape=shape)
    return tf.get_variable(initializer=initializer, name=name)


def conv2d(x, W):
    """
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
