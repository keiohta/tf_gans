# forked from "https://github.com/MingtaoGuo/DCGAN_WGAN_WGAN-GP_LSGAN_SNGAN_RSGAN_RaSGAN_TensorFlow"

import numpy as np
import tensorflow as tf
from PIL import Image

from layers import *
from misc import Logger, immerge


class Generator:
    def __init__(self, name, batch_size, img_size, img_chan):
        self.name = name
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_chan = img_chan

    def __call__(self, inputs):
        with tf.variable_scope(self.name):
            with tf.variable_scope("linear"):
                inputs = tf.reshape((tf.nn.relu((fully_connected(inputs, 4*4*512)))), [self.batch_size, 4, 4, 512])
            with tf.variable_scope("deconv1"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 256, 512], [1, 2, 2, 1], [self.batch_size, 8, 8, 256])))
            with tf.variable_scope("deconv2"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 128, 256], [1, 2, 2, 1], [self.batch_size, 16, 16, 128])))
            with tf.variable_scope("deconv3"):
                inputs = tf.nn.relu(instanceNorm(deconv(inputs, [5, 5, 64, 128], [1, 2, 2, 1], [self.batch_size, 32, 32, 64])))
            with tf.variable_scope("deconv4"):
                stride = 1 if self.img_size <= 32 else 2
                inputs = tf.nn.tanh(deconv(inputs, [5, 5, self.img_chan, 64], [1, stride, stride, 1], [self.batch_size, self.img_size, self.img_size, self.img_chan]))
            return inputs

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class Discriminator:
    def __init__(self, name, batch_size, img_size, img_chan):
        self.name = name
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_chan = img_chan

    def __call__(self, inputs, reuse=False, enable_sn=False):
        with tf.variable_scope(self.name, reuse=reuse):
            with tf.variable_scope("conv1"):
                inputs = tf.nn.leaky_relu(conv(inputs, [5, 5, self.img_chan, 64], [1, 2, 2, 1], enable_sn))
            with tf.variable_scope("conv2"):
                inputs = tf.nn.leaky_relu(instanceNorm(conv(inputs, [5, 5, 64, 128], [1, 2, 2, 1], enable_sn)))
            with tf.variable_scope("conv3"):
                inputs = tf.nn.leaky_relu(instanceNorm(conv(inputs, [5, 5, 128, 256], [1, 2, 2, 1], enable_sn)))
            with tf.variable_scope("conv4"):
                inputs = tf.nn.leaky_relu(instanceNorm(conv(inputs, [5, 5, 256, 512], [1, 2, 2, 1], enable_sn)))
            with tf.variable_scope("logits"):
                inputs = tf.layers.flatten(inputs)
            return fully_connected(inputs, 1, enable_sn)

    @property
    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class GAN:
    def __init__(self, gan_type, batch_size, img_size, img_chan):
        self.gan_types = ["DCGAN", "WGAN", "WGAN-GP", "LSGAN", "SNGAN", "RSGAN", "RaSGAN"]
        assert gan_type in self.gan_types, "[error] not implemented gan_type `{}` specified. choose from following.\r\n{}".format(gan_type, self.gan_types)
        self.gan_type = gan_type
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_chan = img_chan
        self.logger = Logger()
        self._init()

    def _init(self):
        self.Z = tf.placeholder(tf.float32, shape=[self.batch_size, 100])
        self.img = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, self.img_chan])
        D = Discriminator("Discriminator", self.batch_size, self.img_size, self.img_chan)
        G = Generator("Generator", self.batch_size, self.img_size, self.img_chan)

        # with tf.variable_scope(self.gan_type):
        self.fake_img = G(self.Z)
        eps = 1e-14
        if self.gan_type is "DCGAN":
            self.fake_logit = tf.nn.sigmoid(D(self.fake_img))
            self.real_logit = tf.nn.sigmoid(D(self.img, reuse=True))
            self.d_loss = - (tf.reduce_mean(tf.log(self.real_logit + eps)) + tf.reduce_mean(tf.log(1 - self.fake_logit + eps)))
            self.g_loss = - tf.reduce_mean(tf.log(self.fake_logit + eps))
            self.opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(self.g_loss, var_list=G.var)
        elif self.gan_type == "WGAN":
            #WGAN, paper: Wasserstein GAN
            self.fake_logit = D(self.fake_img)
            self.real_logit = D(self.img, reuse=True)
            self.d_loss = - (tf.reduce_mean(self.real_logit) - tf.reduce_mean(self.fake_logit))
            self.g_loss = - tf.reduce_mean(self.fake_logit)
            self.clip = []
            for _, var in enumerate(D.var):
                self.clip.append(tf.clip_by_value(var, -0.01, 0.01))
            self.opt_D = tf.train.RMSPropOptimizer(5e-5).minimize(self.d_loss, var_list=D.var)
            self.opt_G = tf.train.RMSPropOptimizer(5e-5).minimize(self.g_loss, var_list=G.var)
        else:
            raise NotImplementedError
        # statistics
        with tf.variable_scope("statictics"):
            self.summaries = [
                tf.summary.scalar("d_loss", self.d_loss),
                tf.summary.scalar("g_loss", self.g_loss)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.logger.log_graph(sess=self.sess)

    def __call__(self, dataset, n_epoch, test_batch_interval):
        saver = tf.train.Saver()

        print("[info] start training")
        n_trained_step = 0
        for epoch in range(n_epoch):
            for _ in range(dataset.shape[0]//self.batch_size-1):
                idx = n_trained_step // dataset.shape[0]
                batch = dataset[idx:idx+self.batch_size]
                n_trained_step += self.batch_size

                z = np.random.standard_normal([self.batch_size, 100])
                d_loss, g_loss, *summaries = self.sess.run(
                    [self.d_loss, self.g_loss] + self.summaries,
                    feed_dict={self.img: batch, self.Z: z})
                self.logger.write_tf_summary(summaries, n_trained_step)
                print("[info] epoch: {0: 4}, step: {1: 7}, d_loss: {2: 8.4f}, g_loss: {3: 8.4f}".format(epoch, n_trained_step, d_loss, g_loss))

                # update discriminator
                self.sess.run(self.opt_D, feed_dict={self.img: batch, self.Z: z})
                if self.gan_type is "WGAN":
                    self.sess.run(self.clip)
                # update generator
                self.sess.run(self.opt_G, feed_dict={self.img: batch, self.Z: z})
                # test
                if (n_trained_step / self.batch_size) % test_batch_interval == 0:
                    self.test(batch, n_trained_step)

            print("[info] epoch: {0: 4}, step: {1: 7}, d_loss: {2: 8.4f}, g_loss: {3: 8.4f}".format(epoch, n_trained_step, d_loss, g_loss))
            self.test(batch, n_trained_step)
        saver.save(self.sess, self.logger.dir+"/{0:07}_model.ckpt".format(n_trained_step))

    def test(self, batch, n_trained_step):
        z = np.random.standard_normal([self.batch_size, 100])
        imgs = self.sess.run(self.fake_img, feed_dict={self.img: batch, self.Z: z})
        Image.fromarray(immerge(imgs)).save("{}/{:06}.jpg".format(self.logger.dir, n_trained_step))
