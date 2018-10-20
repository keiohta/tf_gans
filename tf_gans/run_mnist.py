import argparse
import numpy as np
import tensorflow as tf

from gans import GAN
from layers import fully_connected


def get_mnist_data():
    print("[info] loading mnist image...")
    mnist = tf.keras.datasets.mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    xs = np.concatenate((x_train, x_test), axis=0)  # 70000x28x28
    xs = np.pad(xs, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.)  # 70000x32x32
    xs = np.expand_dims(xs, axis=xs.ndim)  # 70000x32x32x1
    return xs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan-type', type=str, default="WGAN")
    parser.add_argument('--test-batch-interval', type=int, default=100)
    args = parser.parse_args()

    def discriminator_fn():
        def func(name, inputs, batch_size, img_size, img_chan, enable_sn=False, reuse=False):
            with tf.variable_scope(name, reuse=reuse):
                inputs = tf.layers.flatten(inputs)
                with tf.variable_scope("fc1"):
                    inputs = tf.nn.relu(fully_connected(inputs, 1024, enable_sn))
                with tf.variable_scope("fc2"):
                    inputs = tf.nn.relu(fully_connected(inputs, 1024, enable_sn))
                return fully_connected(inputs, 1, enable_sn)
        return func

    def generator_fn():
        def func(name, inputs, batch_size, img_size, img_chan):
            with tf.variable_scope(name):
                with tf.variable_scope("fc1"):
                    inputs = tf.nn.relu(fully_connected(inputs, 1024))
                with tf.variable_scope("fc2"):
                    inputs = tf.nn.relu(fully_connected(inputs, 1024))
                with tf.variable_scope("output"):
                    inputs = tf.nn.tanh(fully_connected(inputs, img_size*img_size*img_chan))
                    inputs = tf.reshape(inputs, [batch_size, img_size, img_size, img_chan])
                return inputs
        return func

    dataset = get_mnist_data()
    gan = GAN(gan_type=args.gan_type, batch_size=64,
              img_size=int(dataset.shape[1]), img_chan=dataset.shape[3],
              discriminator_fn=discriminator_fn(), generator_fn=generator_fn())
    gan(dataset, n_epoch=100, test_batch_interval=args.test_batch_interval)


if __name__ == "__main__":
    main()
