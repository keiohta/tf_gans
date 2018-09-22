import os
import numpy as np
import datetime
import dateutil.tz
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chainerrl.experiments.prepare_output_dir import prepare_output_dir


class Logger:
    def __init__(
            self,
            args={},
            default_dir='./results',
            specified_dir=None,
            time_format='%Y%m%dT%H%M%S.%f'):
        if specified_dir is not None:
            assert os.path.exists(specified_dir), '[error] specified directory does not exist'
            self._save_dir = specified_dir
        else:
            self._save_dir = prepare_output_dir(args=args, user_specified_dir=default_dir, time_format=time_format)
        self._fig, self._axis = plt.subplots()
        self._tf_writer = tf.summary.FileWriter(self._save_dir)
        print("[info] log directory is '{}'".format(self._save_dir))

    def log_graph(self, sess):
        assert self._tf_writer is not None, "[error] tensorflow filewriter is not set"
        self._tf_writer.add_graph(sess.graph)

    def write_tf_summary(self, summaries, step):
        if not isinstance(summaries, list):
            summaries = [summaries, ]
        if self._tf_writer is not None:
            [self._tf_writer.add_summary(summary, step) for summary in summaries]
        else:
            print("[warn] tensorflow writer is not set to logger")

    @property
    def dir(self):
        return self._save_dir


def mapping(x):
    max = np.max(x)
    min = np.min(x)
    return (x - min) * 255.0 / (max - min + 1e-14)


def immerge(imgs, row=8, col=8):
    """
    merge images into an image with (row * h) * (col * w)
    `imgs` is in shape of N * H * W (* C=1 or 3)
    """
    h, w = imgs.shape[1], imgs.shape[2]
    if imgs.ndim == 4:
        img = np.zeros((h * row, w * col, imgs.shape[3]))
    elif imgs.ndim == 3:
        img = np.zeros((h * row, w * col))

    for idx, _img in enumerate(imgs):
        i = idx % col
        j = idx // col
        img[j * h:j * h + h, i * w:i * w + w, ...] = np.uint8(mapping(_img))
    return np.uint8(img[:, :, 0]) if imgs.shape[3] == 1 else np.uint8(img)
