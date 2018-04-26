import config as cfg
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope


def conv_layer(input, filter, kernel, stride=1, layer_name='Conv'):
    """
    实现卷层
    """
    with tf.name_scope(layer_name):
        net = tf.layers.conv2d(inputs=input, filters=filter, kernel=kernel,
                               stride=stride, padding='SAME', name="Conv")
        return net


def global_average_pooling(x, layer_name='global_average_pooling'):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    with tf.name_scope(layer_name):
        return tf.layers.average_pooling2d(inputs=x, pool_size=[width, height], strides=1,
                                           name="GAP")


def batch_normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, name='drop_out')


def relu(x):
    return tf.nn.relu(x)


def average_pooling(x, pool_size=[2, 2], stride=2, padding="VALID"):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size,
                                       strides=stride, padding=padding)



