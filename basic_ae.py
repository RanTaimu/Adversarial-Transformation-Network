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
            'W_conv1': ne.weight_variable([3, 3, 1, 48], name='W_conv1'),
            'W_conv2': ne.weight_variable([3, 3, 48, 48], name='W_conv2'),
            'W_conv3': ne.weight_variable([3, 3, 48, 48], name='W_conv3'),
            'W_conv4': ne.weight_variable([3, 3, 48, 48], name='W_conv4'),
            'W_conv5': ne.weight_variable([3, 3, 48, 48], name='W_conv5'),
            'W_conv6': ne.weight_variable([3, 3, 48, 48], name='W_conv6'),
            'W_conv7': ne.weight_variable([3, 3, 48, 48], name='W_conv7'),
            'W_conv8': ne.weight_variable([1, 1, 48, 48], name='W_conv8'),
            'W_conv9': ne.weight_variable([1, 1, 48, 1], name='W_conv9')
        }
        return _weights

    @lazy_property
    def biases(self):
        _biases = {
            'b_conv1': ne.bias_variable([48], name='b_conv1'),
            'b_conv2': ne.bias_variable([48], name='b_conv2'),
            'b_conv3': ne.bias_variable([48], name='b_conv3'),
            'b_conv4': ne.bias_variable([48], name='b_conv4'),
            'b_conv5': ne.bias_variable([48], name='b_conv5'),
            'b_conv6': ne.bias_variable([48], name='b_conv6'),
            'b_conv7': ne.bias_variable([48], name='b_conv7'),
            'b_conv8': ne.bias_variable([48], name='b_conv8'),
            'b_conv9': ne.bias_variable([1], name='b_conv9')
        }
        return _biases

    @lazy_property
    def prediction(self):
        h_conv1 = tf.nn.relu(
            ne.conv2d(self.data, self.weights['W_conv1']) +
            self.biases['b_conv1'])
        h_conv2 = tf.nn.relu(
            ne.conv2d(h_conv1, self.weights['W_conv2']) +
            self.biases['b_conv2'])
        h_conv3 = tf.nn.relu(
            ne.conv2d(h_conv2, self.weights['W_conv3']) +
            self.biases['b_conv3'])
        h_conv4 = tf.nn.relu(
            ne.conv2d(h_conv3, self.weights['W_conv4']) +
            self.biases['b_conv4'])
        h_conv5 = tf.nn.relu(
            ne.conv2d(h_conv4, self.weights['W_conv5']) +
            self.biases['b_conv5'])
        h_conv6 = tf.nn.relu(
            ne.conv2d(h_conv5, self.weights['W_conv6']) +
            self.biases['b_conv6'])
        h_conv7 = tf.nn.relu(
            ne.conv2d(h_conv6, self.weights['W_conv7']) +
            self.biases['b_conv7'])
        h_conv8 = tf.nn.relu(
            ne.conv2d(h_conv7, self.weights['W_conv8']) +
            self.biases['b_conv8'])
        h_conv9 = tf.nn.tanh(
            ne.conv2d(h_conv8, self.weights['W_conv9']) +
            self.biases['b_conv9'])

        return tf.reshape(h_conv9, [-1, 784])

    def load(self, sess, path, name='basic_ae.ckpt'):
        saver = tf.train.Saver(dict(self.weights, **self.biases))
        saver.restore(sess, path+'/'+name)

    def save(self, sess, path, name='basic_ae.ckpt'):
        saver = tf.train.Saver(dict(self.weights, **self.biases))
        saver.save(sess, path+'/'+name)
