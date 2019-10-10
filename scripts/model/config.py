import tensorflow as tf
from utils import tf_layer


cfg = {
    'test_fname':r'/home/haha/Documents/PythonPrograms/datasets/ringdata_06_test.tar',
    'train_fname':r'/home/haha/Documents/PythonPrograms/datasets/ringdata_06_train.tar',
    # 'train_fname':r'/home/aiotgo/Documents/WTO/datasets/ringdata_07_train_15993j.tar',
    'log_dir':'log',
    'value_dim':3,
    'batch_size': 128,
    'reg': 0.001,
    'momentum': 0.9,
    'dims': (32, 32, 32),
    'n_channels': 1,
    'n_classes': 10,
    'batches_per_chunk': 64,
    'max_epochs': 100,
    'max_jitter_ij': 2,
    'max_jitter_k': 2,
    'n_rotations': 12,
    'checkpoint_every_nth': 4000,

    'optimizer':'adam',
    'base_learning_rate':0.003,
    'decay_step':1000000,
    'decay_rate':0.5,

    'n_rings':99,
    'n_levels':15,

       }
'''
ARGS
    model:  5D-tensor B*DIMS(DIMS:32*32*32)
'''
NOTE='Model shape: %s * %s * %s , Network : 4conv(256) 1mp(256) 4fc '%(cfg['n_levels'],cfg['n_rings'],cfg['value_dim'])

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
                          padding='VALID', stride=[1, 3], bn=True, is_training=is_training,
                          scope='conv1', bn_decay=bn_decay,weight_decay=weight_decay,
                          activation_fn=tf_layer.leaky_relu)
    # print 'Shape after conv1 :',net.shape
    # TODO revise reshape: we don't need reshape
    # net=tf.reshape(net,(batch_size,n_levels,n_rings,-1))
    # print 'Shape after reshape1 :', net.shape

    net = tf_layer.conv2d(net, 64, [1,1],
                          padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay,weight_decay=weight_decay,
                          activation_fn=tf_layer.leaky_relu)

    net = tf_layer.conv2d(net, 128, [1,1],
                          padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                          scope='conv3', bn_decay=bn_decay,weight_decay=weight_decay,
                          activation_fn=tf_layer.leaky_relu)
    # print 'Shape after conv2 :', net.shape
    net = tf_layer.conv2d(net, 256, [1,1],
                          padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                          scope='conv4', bn_decay=bn_decay,weight_decay=weight_decay,
                          activation_fn=tf_layer.leaky_relu)


    # shape=B*L - R - 1 - D ==> B*L - R - D - 1
    # 10.9 change
    # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    net=tf.squeeze(net)
    net = tf.expand_dims(net, -1)
    net = tf_layer.max_pool2d(net, kernel_size=[1,256],stride=[1,256] , scope='pool1',padding='VALID')
    # net = tf_layer.conv2d(net, 1, [1, 256],
    #                      padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
    #                      scope='conv7', bn_decay=bn_decay, weight_decay=weight_decay,
    #                      activation_fn=tf_layer.leaky_relu)

    # net=tf.transpose(net,perm=[0,1,2,4,3])
    # net=tf.nn.max_pool(net,ksize=[1, 1, 1, 512],strides=[1, 1, 1, 512],padding='VALID',
    #                     name='pool1')


    net = tf.reshape(net, (batch_size, -1))
    net = tf_layer.fully_connected(net, 1024, bn=True, is_training=is_training,
                                   scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_layer.fully_connected(net, 512, bn=True, is_training=is_training,
                                   scope='fc2', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_layer.fully_connected(net, 128, bn=True, is_training=is_training,
                                   scope='fc3', bn_decay=bn_decay,weight_decay=weight_decay)
    net = tf_layer.fully_connected(net, n_classes, bn=True, is_training=is_training,
                                   scope='fc4', bn_decay=bn_decay,weight_decay=weight_decay)

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
    return losses

