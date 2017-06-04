import tensorflow as tf
import util.net_element as ne
from util.decorator import lazy_property


class BasicAE:
    """
    An basic AutoEncoder for mnist.
    """

    def __init__(self, data):
        self.data = tf.reshape(data, [-1, 28, 28, 1])
        self.weights
        self.biases
        self.prediction

    @lazy_property
    def weights(self):
        _weights = {
            'W_conv1': ne.weight_variable([3, 3, 1, 32], name='W_conv1'),
            'W_conv2': ne.weight_variable([3, 3, 32, 64], name='W_conv2'),
            'W_conv3': ne.weight_variable([3, 3, 64, 32], name='W_conv3'),
            'W_conv4': ne.weight_variable([3, 3, 32, 1], name='W_conv4')
        }
        return _weights

    @lazy_property
    def biases(self):
        _biases = {
            'b_conv1': ne.bias_variable([32], name='b_conv1'),
            'b_conv2': ne.bias_variable([64], name='b_conv2'),
            'b_conv3': ne.bias_variable([32], name='b_conv3'),
            'b_conv4': ne.bias_variable([1], name='b_conv4')
        }
        return _biases

    @lazy_property
    def prediction(self):
        h_conv1 = ne.conv2d(self.data, self.weights['W_conv1']) + \
                  self.biases['b_conv1']
        h_conv2 = ne.conv2d(h_conv1, self.weights['W_conv2']) + \
                  self.biases['b_conv2']
        h_conv3 = ne.conv2d(h_conv2, self.weights['W_conv3']) + \
                  self.biases['b_conv3']
        h_conv4 = ne.conv2d(h_conv3, self.weights['W_conv4']) + \
                  self.biases['b_conv4']
        return tf.reshape(h_conv4, [-1, 784])

    def load(self, sess, path, name='basic_ae.ckpt'):
        saver = tf.train.Saver(dict(self.weights, **self.biases))
        saver.restore(sess, path+'/'+name)

    def save(self, sess, path, name='basic_ae.ckpt'):
        saver = tf.train.Saver(dict(self.weights, **self.biases))
        saver.save(sess, path+'/'+name)
