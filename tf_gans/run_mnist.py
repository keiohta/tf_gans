import numpy as np
import tensorflow as tf

from gans import GAN


def get_mnist_data():
    print("[info] loading mnist image...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    xs = np.concatenate((x_train, x_test), axis=0)
    xs = np.pad(xs, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.)
    xs = np.expand_dims(xs, axis=xs.ndim)
    dataset = tf.data.Dataset.from_tensor_slices(xs)
    print("[info] done. shape: {}".format(dataset.output_shapes))
    return dataset


def main():
    dataset = get_mnist_data()
    gan = GAN(gan_type="DCGAN", batch_size=64, img_size=int(dataset.output_shapes[0]), img_chan=1)
    gan(dataset, n_epoch=100)


if __name__ == "__main__":
    main()
