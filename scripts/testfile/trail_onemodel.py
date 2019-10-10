import tensorflow as tf
from utils import tf_layer


cfg = {'batch_size': 32,
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
       'base_learning_rate':0.002,
       'decay_step':500000,
       'decay_rate':0.5,

       'n_rings':99,
       'n_levels':15,

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
    n_rings,n_levels= cfg['n_rings'], cfg['n_levels']

    # TODO pack the activations into layers
    input_model=tf.reshape(model,(-1,n_rings,3))
    input_model = tf.expand_dims(input_model, -1)
    net = tf_layer.conv2d(input_model, 64, [1,3],
                          padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                          scope='conv1', bn_decay=bn_decay,weight_decay=weight_decay,activation_fn=tf_layer.leaky_relu)
    # print 'Shape after conv1 :',net.shape
    # TODO revise reshape: we don't need reshape
    # net=tf.reshape(net,(batch_size,n_levels,n_rings,-1))
    # print 'Shape after reshape1 :', net.shape

    # net = tf_layer.conv2d(net, 64, [1,1],
    #                       padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
    #                       scope='conv2', bn_decay=bn_decay,weight_decay=weight_decay)
    # net = tf_layer.leaky_relu(net)

    net = tf_layer.conv2d(net, 64, [1,1],
                          padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay,weight_decay=weight_decay,
                          activation_fn=tf_layer.leaky_relu)
    net = tf_layer.conv2d(net, 128, [1,1],
                          padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                          scope='conv3', bn_decay=bn_decay,weight_decay=weight_decay,
                          activation_fn=tf_layer.leaky_relu)
    # print 'Shape after conv2 :', net.shape
    net = tf_layer.conv2d(net, 512, [1,1],
                          padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                          scope='conv4', bn_decay=bn_decay,weight_decay=weight_decay,
                          activation_fn=tf_layer.leaky_relu)
    # print 'Shape after conv3 :', net.shape
    net = tf.reshape(net, (-1, n_rings, 512))
    net = tf.expand_dims(net, -1)
    net = tf_layer.max_pool2d(net, kernel_size=[1,512],stride=[1,512] , scope='pool1',padding='VALID')

    # net=tf.transpose(net,perm=[0,1,2,4,3])
    # net=tf.nn.max_pool(net,ksize=[1, 1, 1, 512],strides=[1, 1, 1, 512],padding='VALID',
    #                     name='pool1')
    net = tf.reshape(net, (batch_size*n_levels, -1))


    net = tf_layer.fully_connected(net, 128, bn=True, is_training=is_training,
                                   scope='fc1', bn_decay=bn_decay,weight_decay=weight_decay)
    net = tf_layer.fully_connected(net, 1, bn=True, is_training=is_training,
                                   scope='fc2', bn_decay=bn_decay,weight_decay=weight_decay)
    net = tf.reshape(net, (batch_size, -1))
    net = tf_layer.fully_connected(net, 1024, bn=True, is_training=is_training,
                                   scope='fc3', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_layer.fully_connected(net, 512, bn=True, is_training=is_training,
                                   scope='fc4', bn_decay=bn_decay,weight_decay=weight_decay)
    net = tf_layer.fully_connected(net, 128, bn=True, is_training=is_training,
                                   scope='fc5', bn_decay=bn_decay,weight_decay=weight_decay)
    net = tf_layer.fully_connected(net, n_classes, bn=True, is_training=is_training,
                                   scope='fc6', bn_decay=bn_decay,weight_decay=weight_decay)
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

