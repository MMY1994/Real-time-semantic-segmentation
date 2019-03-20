import tensorflow as tf
from layers.pooling import max_pool_2d
from layers.utils import *

def depthwise_separable_conv2d(name, x, w_depthwise=None, w_pointwise=None, width_multiplier=1.0, num_filters=16,
                               kernel_size=(3, 3), t = None, 
                               padding='SAME', stride=(1, 1),
                               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, biases=(0.0, 0.0),
                               activation=None, batchnorm_enabled=True,
                               is_training=True):
    total_num_filters = int(round(num_filters * width_multiplier))
    expansion_num_filters = int(x.get_shape()[-1] * t)
    with tf.variable_scope(name) as scope:
        conv_b = conv2d('pointwise1', x=x, w=w_pointwise, num_filters=expansion_num_filters, kernel_size=(1, 1),
                        initializer=initializer, l2_strength=l2_strength, bias=biases[1], activation=activation,
                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)

        conv_a = depthwise_conv2d('depthwise', x=conv_b, w=w_depthwise, kernel_size=kernel_size, padding=padding,
                                  stride=stride,
                                  initializer=initializer, l2_strength=l2_strength, bias=biases[0],
                                  activation=activation,
                                  batchnorm_enabled=batchnorm_enabled, is_training=is_training)

        conv_o = conv2d('pointwise2', x=conv_a, w=w_pointwise, num_filters=total_num_filters, kernel_size=(1, 1),
                        initializer=initializer, l2_strength=l2_strength, bias=biases[1], activation=None,
                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)

        if stride[0] == 1:
           ins = conv2d('ins', x=x, w=w_pointwise, num_filters=total_num_filters, kernel_size=(1, 1),
                        initializer=initializer, l2_strength=l2_strength, bias=biases[1], activation=activation,
                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)
           conv_c = conv_o + ins
        else:
           conv_c = conv_o

    return conv_c

def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) 由tf.name_scope('name') as scope提供.
    :param x: (tf.tensor)输入(N, H, W, C).
    :param num_filters: (integer) No. of filters (输出的深度)
    :param kernel_size: (integer tuple) 卷积核的大小.
    :param padding: (string) padding的大小.
    :param stride: (integer tuple) 步长.
    :param initializer: (tf.contrib initializer) 初始化, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 正则化参数.
    :param bias: (float) 偏置.
    :param activation: (tf.graph operator) 卷积运算后的激活函数. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) 保持多少个神经元的概率. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_p(scope, x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o


def depthwise_conv2d(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True):
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_p(name=scope, x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    return conv_a



def __depthwise_conv2d_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out

def __conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out


