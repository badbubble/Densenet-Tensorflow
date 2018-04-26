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
        net = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel,
                               strides=stride, padding='SAME')
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
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def relu(x):
    return tf.nn.relu(x)


def average_pooling(x, pool_size=[2, 2], stride=2, padding="VALID"):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size,
                                       strides=stride, padding=padding)


def max_pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def concatenation(layers):
    return tf.concat(layers, axis=3)


def linear(x):
    return tf.layers.dense(inputs=x, units=cfg.class_num, name='linear')


class DenseNet(object):
    def __init__(self, x,  training=True):
        self.nb_blocks = cfg.nb_block
        self.filters = cfg.growth_rate
        self.training = training
        self.model = self.densenet(x)

    def bottleneck_layer(self, x, scope):
        """
        BN->Relu->Conv(1, 1)->BN->Relu->Conv(3, 3)
        要产生4k个feature map
        """
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope+'_nb1')
            x = relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1],
                           layer_name=scope+'_conv1')
            x = batch_normalization(x, training=self.training, scope=scope+'_nb2')
            x = relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope+'_conv2')
            x = drop_out(x, rate=cfg.dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        """
        论文中对应pooling layer: BN->Conv(1, 1)->AP(2, 2)
        """
        with tf.name_scope(scope):
            x = batch_normalization(x, training=self.training, scope=scope+'_bn1')
            #  relu论文中没有
            x = relu(x)
            #
            x = conv_layer(input=x, filter=self.filters, kernel=[1, 1],
                           layer_name=scope+'_conv1')
            x = drop_out(x, rate=cfg.dropout_rate, training=self.training)
            x = average_pooling(x, pool_size=[2, 2], stride=2)
            return x

    def dense_block(self, x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layer_concat = list()
            layer_concat.append(x)

            x = self.bottleneck_layer(x, scope=layer_name+'_bottle_'+str(0))

            layer_concat.append(x)
            for i in range(nb_layers-1):
                x = concatenation(layer_concat)
                x = self.bottleneck_layer(x, scope=layer_name+'_bottle_'+str(i+1))
                layer_concat.append(x)

            x = concatenation(x)
            return x

    def densenet(self, x):
        """
        先要有卷积和池化
        """
        x = conv_layer(input=x, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='Conv0')
        x = max_pooling(x, pool_size=[3, 3], stride=2)

        # for i in range(self.nb_blocks):
        #     x = self.dense_block(x, 4, layer_name='dense_'+str(i))
        #     x = self.transition_layer(x, scope='trans_' + str(i))
        x = self.dense_block(x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')
        x = self.dense_block(x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')
        # x = self.dense_block(x=x, nb_layers=48, layer_name='dense_3')
        # x = self.transition_layer(x, scope='trans_3')
        # x = self.dense_block(x=x, nb_layers=32, layer_name='dense_final')

        x = batch_normalization(x, training=self.training, scope='linear_bn')
        x = relu(x)
        x = global_average_pooling(x)
        x = tf.layers.flatten(x)
        x = linear(x)

        return x
