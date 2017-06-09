import tensorflow as tf
import numpy as np
import atn_model as atn
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('train', False, 'Train and save the ATN model.')

# Placeholder nodes.
images_holder = tf.placeholder(tf.float32, [None, 784])
label_holder = tf.placeholder(tf.float32, [None, 10])
p_keep_holder = tf.placeholder(tf.float32)
rerank_holder = tf.placeholder(tf.float32, [None, 10])

mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)


def main(arvg=None):
    """
    """
    if FLAGS.train:
        train()
    else:
        test()


def test():
    """
    """
    model = atn.ATN(images_holder, label_holder, p_keep_holder, rerank_holder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load(sess, './Models/AE_for_ATN')

        adv_images = sess.run(
            model.prediction,
            feed_dict={images_holder: mnist.test.images}
        )

        print('Original accuracy: {0:0.5f}'.format(
            sess.run(model._target.accuracy, feed_dict={
                images_holder: mnist.test.images,
                label_holder: mnist.test.labels,
                p_keep_holder: 1.0
            })))

        print('Attacked accuracy: {0:0.5f}'.format(
            sess.run(model._target.accuracy, feed_dict={
                images_holder: adv_images,
                label_holder: mnist.test.labels,
                p_keep_holder: 1.0
            })))

        # Show some results.
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(adv_images[i], (28, 28)))
        plt.show()


def train():
    """
    """
    attack_target = 8
    alpha = 1.5
    training_epochs = 10
    batch_size = 64

    model = atn.ATN(images_holder, label_holder, p_keep_holder, rerank_holder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model._target.load(sess, './Models/AE_for_ATN/BasicCNN')

        total_batch = int(mnist.train.num_examples/batch_size)
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                r_res = sess.run(model._target.prediction,
                                 feed_dict={
                                     images_holder: batch_xs,
                                     p_keep_holder: 1.0
                                 })
                r_res[:, attack_target] = np.max(r_res, axis=1) * alpha
                norm_div = np.linalg.norm(r_res, axis=1)
                for i in range(len(r_res)):
                    r_res[i] /= norm_div[i]

                _, loss = sess.run(model.optimization, feed_dict={
                    images_holder: batch_xs,
                    p_keep_holder: 1.0,
                    rerank_holder: r_res
                })

            print('Eopch {0} completed. loss = {1}'.format(epoch+1, loss))
        print("Optimization Finished!")

        model.save(sess, './Models/AE_for_ATN')
        print("Trained params have been saved to './Models/AE_for_ATN'")


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()
