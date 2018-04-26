import config as cfg
import tensorflow as tf
import numpy as np


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

