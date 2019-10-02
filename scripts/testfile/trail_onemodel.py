import tensorflow as tf
import numpy as np
from model import config
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from utils import npytar

CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True
CONFIG.allow_soft_placement = True
CONFIG.log_device_placement = False
GPU_INDEX =0
LOG_DIR = 'log'
cfg = config.cfg
dims = cfg['dims']
max_epochs=cfg['max_epochs']
fname=r'/home/haha/Documents/PythonPrograms/datasets/shapenet10_train.tar'

def jitter_chunk(src):
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, :, ::-1, :] = dst
    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+2)
    return dst


def data_loader(fname):

    dims = cfg['dims']
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        return x,int(name.split('.')[0])-1




def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            batch_index = tf.placeholder(tf.int32, name='batch_index')  # batch_index = T.iscalar('batch_index')
            X_pl = tf.placeholder(tf.float32, shape=[32,32,32 ], name='X')  # X = T.TensorType('float32', [False]*5)('X')
            y_pl = tf.placeholder(tf.float32, shape=[None, ], name='y')  # y = T.TensorType('int32', [False]*1)('y')
            is_training_pl = tf.placeholder(tf.bool, shape=())
            batch_slice = slice(batch_index * cfg['batch_size'], (batch_index + 1) * cfg['batch_size'])
            yhat=config.get_model(model=X_pl,is_training=is_training_pl,bn_decay=None)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        sess = tf.Session(config=CONFIG)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'X_pl': X_pl,
               # 'y_pl': y_pl,
               'is_training_pl': is_training_pl,
               }

        x,name=data_loader(fname)
        feed_dict = {ops['X_pl']:x ,
                     ops['is_training_pl']: True, }

        print sess.run(yhat,feed_dict=feed_dict)




def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    loader = (data_loader(cfg, fname))
    for X, y in loader:
        num_batches = len(X) // cfg['batch_size']
        feed_dict = {ops['X_pl']: X,
                     ops['y_pl']: y,
                     ops['is_training_pl']: is_training, }


