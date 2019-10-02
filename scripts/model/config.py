import tensorflow as tf
from utils import tf_layer

lr_schedule = {0: 0.001,
               60000: 0.0001,
               400000: 0.00005,
               600000: 0.00001,
               }

cfg = {'batch_size': 32,
       'learning_rate': lr_schedule,
       'reg': 0.001,
       'momentum': 0.9,
       'dims': (32, 32, 32),
       'n_channels': 1,
       'n_classes': 10,
       'batches_per_chunk': 64,
       'max_epochs': 80,
       'max_jitter_ij': 2,
       'max_jitter_k': 2,
       'n_rotations': 12,
       'checkpoint_every_nth': 4000,

       'optimizer':'momentum',
       'base_learning_rate':0.001,
       'decay_step':200000,
       'decay_rate':0.1,

       }
'''
ARGS
    model:  5D-tensor B*DIMS(DIMS:32*32*32)
'''


def get_model(model, is_training, bn_decay=None):
    '''change into B*DIMS*C'''
    n_classes=cfg['n_classes']
    batch_size = model.get_shape()[0].value
    weight_decay=cfg['reg'] # L2 reg

    input_model = tf.expand_dims(model, -1)
    net = tf_layer.conv3d(input_model, 32, [5, 5, 5],
                          padding='VALID', stride=[1, 1,1], bn=True, is_training=is_training,
                          scope='conv1', bn_decay=bn_decay,weight_decay=weight_decay)
    net = tf_layer.conv3d(net, 32, [3, 3, 3],
                          padding='VALID', stride=[1, 1,1], bn=True, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay,weight_decay=weight_decay)
    net = tf_layer.max_pool3d(net, [2, 2, 2], padding='VALID', scope='pool1')
    net = tf.reshape(net, (batch_size, -1))
    net = tf_layer.fully_connected(net, 128, bn=True, is_training=is_training,
                                   scope='fc1', bn_decay=bn_decay,weight_decay=weight_decay)
    net = tf_layer.fully_connected(net, n_classes, bn=True, is_training=is_training,
                                   scope='fc2', bn_decay=bn_decay,weight_decay=weight_decay)

    return net

''' cross entropy loss with L2 regularization'''
def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.add_to_collection('losses', classify_loss)
    losses=tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('classify loss', classify_loss)
    tf.summary.scalar('total loss (with l2-reg)', losses)
    return losses

