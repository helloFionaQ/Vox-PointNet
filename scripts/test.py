import time

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
from utils import class2name
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cfg = config.cfg
MODEL_FILE = os.path.join(BASE_DIR,'model', 'config.py')

BATCH_SIZE=cfg['batch_size']
GPU_INDEX =0
MODEL_PATH ='log/model.ckpt'
DUMP_DIR = 'log/test'
model=importlib.import_module('model.config')

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_test.txt'), 'w')

dims, max_epochs, batch_size, classes = cfg['dims'], cfg['max_epochs'], cfg['batch_size'], cfg['n_classes']
name_dic = class2name.class_id_to_name
n_levels, n_rings=cfg['n_levels'], cfg['n_rings']

fname=r'/home/haha/Documents/PythonPrograms/datasets/ringdata_03_test.tar'


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def data_loader(fname):
    dims = cfg['dims']
    chunk_size = cfg['n_rotations']
    xc = np.zeros((chunk_size,n_levels,n_rings,3), dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            yield (xc, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    assert(len(yc)==0)


def evaluate(num_votes):


    with tf.device('/gpu:' + str(GPU_INDEX)):
        chunk_size = cfg['n_rotations']

        X_pl = tf.placeholder(tf.float32, shape=[chunk_size,n_levels,n_rings,3], name='X')
        y_pl = tf.placeholder(tf.int32, shape=[None, ], name='y')  # y = T.TensorType('int32', [False]*1)('y')
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Get model
        out = model.get_model(X_pl, is_training_pl)
        loss = model.get_loss(out, y_pl)  # float32
        pred = tf.argmax(out, 1)
        correct = tf.equal(pred, tf.to_int64(y_pl))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # error_rate = T.cast( T.mean( T.neq(pred, y) ), 'float32' )
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'X_pl': X_pl,
           'y_pl': y_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'accuracy':accuracy,}
    log_string("Start testing.")
    start_time = time.time()

    eval_one_epoch(sess, ops, num_votes)
    end_time = time.time()
    log_string("Total running time: %.3f seconds" % ((end_time - start_time)))


def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    # TODO Shuffle train files
    loader = (data_loader(fname))
    accs = []
    y_gnd=np.zeros(classes,dtype=np.int32)
    y_hat=np.zeros(classes,dtype=np.int32)
    for X, y in loader:
        num_batches = len(X) // cfg['batch_size']

        feed_dict = {ops['X_pl']: X,
                     ops['y_pl']: y,
                     ops['is_training_pl']: is_training, }
        pred_val, accuracy_val = sess.run([ops['pred'], ops['accuracy']],
                                                                      feed_dict=feed_dict)
        for i,yg in enumerate(y):
            yg=int(yg)
            y_gnd[yg]+=1
            if yg==pred_val[i]:
                y_hat[yg]+=1
        accs.append(accuracy_val)

    accs_cls =[]
    for i in range(classes):
        accs_cls.append((float(y_hat[i])/float(y_gnd[i])))
    acc_cls=float(np.mean(accs_cls))
    acc =  float(np.mean(accs))
    log_string('accuracy all: %f' % ( acc))
    log_string('accuracy classes: %f' % ( acc_cls))
    log_string('accuracy for each class:' )
    for i in range(classes):
        log_string('   %s : %f'%(name_dic[i+1],(float(y_hat[i])/float(y_gnd[i]))))


        # loss, acc = float(np.mean(lvs)), float(np.mean(acc))
        # print 'epoch: {}, itr: {}, loss: {}, acc: {}'.format(epoch, itr, loss, acc)
        # mlog.log(epoch=epoch, itr=itr, loss=loss, acc=acc)


if __name__ == '__main__':
    # with tf.Graph().as_default():
    #     evaluate(num_votes=1)
    evaluate(num_votes=1)
    LOG_FOUT.close()
