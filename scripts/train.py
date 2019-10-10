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
from utils import npytar, class2name
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cfg = config.cfg
LOG_DIR = cfg['log_dir']
TRAIN_DIR = 'train'
MODEL_FILE = os.path.join(BASE_DIR,'model', 'config.py')
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
OPTIMIZER = 'adam' # adam or momentum
MOMENTUM = cfg['momentum']
BATCH_SIZE=cfg['batch_size']
BASE_LEARNING_RATE=cfg['base_learning_rate']
DECAY_STEP=cfg['decay_step']
DECAY_RATE=cfg['decay_rate']
GPU_INDEX =0
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(200000)
BN_DECAY_CLIP = 0.99

name_dic = class2name.class_id_to_name

dims, max_epochs, batch_size, classes = cfg['dims'], cfg['max_epochs'], cfg['batch_size'], cfg['n_classes']
n_levels, n_rings, value_dim=cfg['n_levels'], cfg['n_rings'],cfg['value_dim']
model=importlib.import_module('model.config')

test_fname=cfg['test_fname']
train_fname=cfg['train_fname']
NOTE=config.NOTE

tf.reset_default_graph()

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
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def jitter_chunk(src):
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, ::-1, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :] = dst
    max_ij = cfg['max_jitter_ij']
    max_k = cfg['max_jitter_k']
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+1)
    return dst

def data_loader(cfg, fname):

    chunk_size = cfg['batch_size']*cfg['batches_per_chunk']
    xc = np.zeros((chunk_size,n_levels,n_rings,value_dim), dtype=np.float32)
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
    if len(yc) > 0:
        # pad to nearest multiple of batch_size

        new_size = int(np.ceil(len(yc)/float(cfg['batch_size'])))*cfg['batch_size']
        xc = xc[:new_size]
        xc[len(yc):] = xc[:(new_size-len(yc))]
        yc = yc + yc[:(new_size-len(yc))]

        yield (xc, np.asarray(yc, dtype=np.float32))

def train():

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            batch_index = tf.Variable(0)  # batch_index = T.iscalar('batch_index')
            X_pl = tf.placeholder(tf.float32, shape=[batch_size,n_levels,n_rings,3], name='X')  # X = T.TensorType('float32', [False]*5)('X')
            y_pl = tf.placeholder(tf.int32, shape=[None, ], name='y')  # y = T.TensorType('int32', [False]*1)('y')
            is_training_pl = tf.placeholder(tf.bool, shape=())
            bn_decay = get_bn_decay(batch_index)

            # Get model and loss
            out=model.get_model(X_pl,is_training_pl)
            # out=model.get_model(X_pl,is_training_pl,  bn_decay=bn_decay)
            loss=model.get_loss(out,y_pl) # float32
            tf.summary.scalar('loss', loss)
            pred=tf.argmax(out, 1)
            correct=tf.equal(pred, tf.to_int64(y_pl))
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
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, TRAIN_DIR),
                                  sess.graph)


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
        log_string(NOTE)
        log_string('Dataset : %s'%os.path.split(train_fname)[-1])
        print 'Training...'
        start_time = time.time()

        for epoch in xrange(max_epochs):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            # eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 5 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
                log_string("Test one epoch:" )
                test_one_epoch(sess, ops)

        log_string("Training finished")
        test_one_epoch(sess, ops)
        end_time = time.time()
        log_string("Total running time: %.3f hours" % ((end_time - start_time) / 60.0 / 60.0))



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    # TODO Shuffle train files
    loader = (data_loader(cfg, train_fname))
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
        loss, acc = float(np.mean(losses)), float(np.mean(accs))
        log_string('itr: %d -- mean loss: %f  accuracy: %f'  %(step , loss ,acc))

def test_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    loader = (data_loader(cfg, test_fname))
    losses, accs = [], []
    y_gnd = np.zeros(classes, dtype=np.int32)
    y_hat = np.zeros(classes, dtype=np.int32)
    for X, y in loader:
        num_batches = len(X) // cfg['batch_size']

        for bi in xrange(num_batches):
            # if bi%8!=7:
            #     continue
            batch_slice = slice(bi * cfg['batch_size'], (bi + 1) * cfg['batch_size'])
            feed_dict = {ops['X_pl']: X[batch_slice],
                         ops['y_pl']: y[batch_slice],
                         ops['is_training_pl']: is_training, }
            step,loss_val, pred_val, accuracy_val = sess.run([ops['step'],ops['loss'], ops['pred'],
                                                                           ops['accuracy']],
                                                            feed_dict=feed_dict)

            losses.append(loss_val)
            accs.append(accuracy_val)
            for i, yg in enumerate(y[batch_slice]):
                yg = int(yg)
                y_gnd[yg] += 1
                if yg == pred_val[i]:
                    y_hat[yg] += 1

    loss, acc = float(np.mean(losses)), float(np.mean(accs))
    tf.summary.scalar('test_accuracy', acc)
    tf.summary.scalar('night_stand_accuracy',(float(y_hat[6]) / float(y_gnd[6])))
    log_string('itr: %d -- mean loss: %f  accuracy: %f'  %(step , loss ,acc))
    log_string('accuracy for each class:')
    for i in range(classes):
        log_string('   %s : %f (%d)' % (name_dic[i + 1], (float(y_hat[i]) / float(y_gnd[i])),y_gnd[i]))






if __name__ == "__main__":
    train()
    LOG_FOUT.close()