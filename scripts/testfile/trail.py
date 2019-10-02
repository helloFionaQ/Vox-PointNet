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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cfg = config.cfg
LOG_DIR = 'log'
MODEL_FILE = os.path.join(BASE_DIR,'model', 'config.py')

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
OPTIMIZER = 'momentum' # adam or momentum
MOMENTUM = cfg['momentum']
BATCH_SIZE=cfg['batch_size']
BASE_LEARNING_RATE=cfg['base_learning_rate']
DECAY_STEP=cfg['decay_step']
DECAY_RATE=cfg['decay_rate']
GPU_INDEX =0



dims = cfg['dims']
model=importlib.import_module('model.config')
max_epochs=cfg['max_epochs']
fname=r'/home/haha/Documents/PythonPrograms/datasets/shapenet10_train.tar'


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

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


def data_loader(cfg, fname):

    dims = cfg['dims']
    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size, cfg['n_channels'],)+dims, dtype=np.float32)
    reader = npytar.NpyTarReader(fname)
    yc = []
    for ix, (x, name) in enumerate(reader):
        cix = ix % chunk_size
        xc[cix] = x.astype(np.float32)
        yc.append(int(name.split('.')[0])-1)
        if len(yc) == chunk_size:
            xc = jitter_chunk(xc, cfg)
            yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))
            yc = []
            xc.fill(0)
    if len(yc) > 0:
        # pad to nearest multiple of batch_size
        if len(yc)%cfg['batch_size'] != 0:
            new_size = int(np.ceil(len(yc)/float(cfg['batch_size'])))*cfg['batch_size']
            xc = xc[:new_size]
            xc[len(yc):] = xc[:(new_size-len(yc))]
            yc = yc + yc[:(new_size-len(yc))]

        xc = jitter_chunk(xc, cfg)
        yield (2.0*xc - 1.0, np.asarray(yc, dtype=np.float32))


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            batch_index = tf.Variable(0)  # batch_index = T.iscalar('batch_index')
            X_pl = tf.placeholder(tf.float32, shape=[None,] *5, name='X')  # X = T.TensorType('float32', [False]*5)('X')
            y_pl = tf.placeholder(tf.int32, shape=[None, ], name='y')  # y = T.TensorType('int32', [False]*1)('y')
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # TODO set bn_decay

            # Get model and loss
            out=model.get_model(X_pl,is_training_pl)
            loss=model.get_loss(out,y_pl) # float32
            tf.summary.scalar('loss', loss)
            pred=tf.argmax(out, 1)
            correct=tf.equal(tf.argmax(out, 1), tf.to_int32(y_pl))
            accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))  # error_rate = T.cast( T.mean( T.neq(pred, y) ), 'float32' )
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch_index)
            tf.summary.scalar('learning_rate', learning_rate)

            # TODO try adam
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            train_op = optimizer.minimize(loss, global_step=batch_index)
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)


        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'X_pl': X_pl,
               'y_pl': y_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'accuracy':accuracy,
               'train_op': train_op,
               'merged': merged,
               'step': batch_index
               }

        for epoch in range(max_epochs):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            # eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    print 'Training...'
    start_time = time.time()
    # TODO Shuffle train files
    for epoch in xrange(cfg['max_epochs']):
        loader = (data_loader(cfg, fname))
        for X, y in loader:
            num_batches = len(X) // cfg['batch_size']
            losses, accs = [], []
            for bi in xrange(num_batches):
                batch_slice = slice(bi * cfg['batch_size'], (bi + 1) * cfg['batch_size'])
                feed_dict = {ops['X_pl']: X[batch_slice],
                             ops['y_pl']: y[batch_slice],
                             ops['is_training_pl']: is_training, }
                summary, step, _, loss_val, pred_val, accuracy_val = sess.run([ops['merged'], ops['step'],
                                                                 ops['train_op'], ops['loss'], ops['pred'],
                                                                               ops['accuracy']],
                                                                feed_dict=feed_dict)
                train_writer.add_summary(summary, step)
                losses.append(loss_val)
                accs.append(accuracy_val)
            loss, acc = float(np.mean(losses)), float(np.mean(acc))
            log_string('mean loss: %f' % loss)
            log_string('accuracy: %f' % acc)
            # loss, acc = float(np.mean(lvs)), float(np.mean(acc))
            # print 'epoch: {}, itr: {}, loss: {}, acc: {}'.format(epoch, itr, loss, acc)
            # mlog.log(epoch=epoch, itr=itr, loss=loss, acc=acc)

    end_time = time.time()
    log_string("Total running time: %.3f hours" % ((end_time - start_time) / 60.0 / 60.0))



