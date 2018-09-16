import numpy as np
import tensorflow as tf


eps = 1e-14


def spectral_norm(name, w, iteration=1):
    # forked from "https://github.com/taki0112/Spectral_Normalization-Tensorflow"
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    with tf.variable_scope(name, reuse=False):
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None

    def l2_norm(v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


def instanceNorm(inputs):
    mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    scale = tf.get_variable("scale", shape=mean.shape, initializer=tf.constant_initializer([1.0]))
    shift = tf.get_variable("shift", shape=mean.shape, initializer=tf.constant_initializer([0.0]))
    return (inputs - mean) * scale / (tf.sqrt(var + eps)) + shift


def deconv(inputs, shape, strides, out_num, enable_sn=False):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-2]], initializer=tf.constant_initializer([0]))
    return tf.nn.conv2d_transpose(inputs, spectral_norm("sn", filters), out_num, strides) + bias if enable_sn \
        else tf.nn.conv2d_transpose(inputs, filters, out_num, strides) + bias


def conv(inputs, shape, strides, enable_sn=False):
    filters = tf.get_variable("kernel", shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))
    bias = tf.get_variable("bias", shape=[shape[-1]], initializer=tf.constant_initializer([0]))
    return tf.nn.conv2d(inputs, spectral_norm("sn", filters), strides, "SAME") + bias if enable_sn \
        else tf.nn.conv2d(inputs, filters, strides, "SAME") + bias


def fully_connected(inputs, n_out, enable_sn=False):
    W = tf.get_variable("W", [inputs.shape[-1], n_out], initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [n_out], initializer=tf.constant_initializer([0]))
    return tf.matmul(inputs, spectral_norm("sn", W)) + b if enable_sn else tf.matmul(inputs, W) + b


def leaky_relu(inputs, slope=0.2):
    return tf.maximum(slope*inputs, inputs)
