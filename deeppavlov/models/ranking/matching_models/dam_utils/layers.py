# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @inproceedings{ ,
#   title={Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network},
#   author={Xiangyang Zhou, Lu Li, Daxiang Dong, Yi Liu, Ying Chen, Wayne Xin Zhao, Dianhai Yu and Hua Wu},
#   booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
#   volume={1},
#   pages={  --  },
#   year={2018}
# }
# ```
# http://aclweb.org/anthology/P18-1103
#
# Based on authors' Tensorflow code: https://github.com/baidu/Dialogue/tree/master/DAM

from logging import getLogger

import tensorflow as tf

import deeppavlov.models.ranking.matching_models.dam_utils.operations as op

log = getLogger(__name__)


def similarity(x, y, x_lengths, y_lengths):
    '''calculate similarity with two 3d tensor.

    Args:
        x: a tensor with shape [batch, time_x, dimension]
        y: a tensor with shape [batch, time_y, dimension]

    Returns:
        a tensor with shape [batch, time_x, time_y]

    Raises:
        ValueError: if
            the dimenisons of x and y are not equal.
    '''
    with tf.variable_scope('x_attend_y'):
        try:
            x_a_y = block(
                x, y, y,
                Q_lengths=x_lengths, K_lengths=y_lengths)
        except ValueError:
            tf.get_variable_scope().reuse_variables()
            x_a_y = block(
                x, y, y,
                Q_lengths=x_lengths, K_lengths=y_lengths)

    with tf.variable_scope('y_attend_x'):
        try:
            y_a_x = block(
                y, x, x,
                Q_lengths=y_lengths, K_lengths=x_lengths)
        except ValueError:
            tf.get_variable_scope().reuse_variables()
            y_a_x = block(
                y, x, x,
                Q_lengths=y_lengths, K_lengths=x_lengths)

    return tf.matmul(x + x_a_y, y + y_a_x, transpose_b=True)


def dynamic_L(x):
    '''Attention machanism to combine the infomation, 
       from https://arxiv.org/pdf/1612.01627.pdf.

    Args:
        x: a tensor with shape [batch, time, dimension]

    Returns:
        a tensor with shape [batch, dimension]

    Raises:
    '''
    key_0 = tf.get_variable(
        name='key',
        shape=[x.shape[-1]],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(
            -tf.sqrt(6. / tf.cast(x.shape[-1], tf.float32)),
            tf.sqrt(6. / tf.cast(x.shape[-1], tf.float32))))

    key = op.dense(x, add_bias=False)  # [batch, time, dimension]
    weight = tf.reduce_sum(tf.multiply(key, key_0), axis=-1)  # [batch, time]
    weight = tf.expand_dims(tf.nn.softmax(weight), -1)  # [batch, time, 1]

    L = tf.reduce_sum(tf.multiply(x, weight), axis=1)  # [batch, dimension]
    return L


def loss(x, y, num_classes=2, is_clip=True, clip_value=10):
    '''From info x calculate logits as return loss.

    Args:
        x: a tensor with shape [batch, dimension]
        num_classes: a number

    Returns:
        loss: a tensor with shape [1], which is the average loss of one batch
        logits: a tensor with shape [batch, 1]

    Raises:
        AssertionError: if
            num_classes is not a int greater equal than 2.
    TODO:
        num_classes > 2 may be not adapted.
    '''
    assert isinstance(num_classes, int)
    assert num_classes >= 2

    # W = tf.get_variable(
    #     name='weights',
    #     shape=[x.shape[-1], num_classes-1],
    #     initializer=tf.orthogonal_initializer())
    # bias = tf.get_variable(
    #     name='bias',
    #     shape=[num_classes-1],
    #     initializer=tf.zeros_initializer())
    #
    # logits = tf.reshape(tf.matmul(x, W) + bias, [-1])
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(
    #     labels=tf.cast(y, tf.float32),
    #     logits=logits)
    # loss = tf.reduce_mean(tf.clip_by_value(loss, -clip_value, clip_value))

    W = tf.get_variable(
        name='weights',
        shape=[x.shape[-1], num_classes],
        initializer=tf.orthogonal_initializer())
    bias = tf.get_variable(
        name='bias',
        shape=[num_classes],
        initializer=tf.zeros_initializer())

    logits = tf.matmul(x, W) + bias
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,
        logits=logits)
    loss = tf.reduce_mean(tf.clip_by_value(loss, -clip_value, clip_value))

    return loss, logits


def attention(
        Q, K, V,
        Q_lengths, K_lengths,
        attention_type='dot',
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add attention layer.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, Q_time, V_dimension]

    Raises:
        AssertionError: if
            Q_dimension not equal to K_dimension when attention type is dot.
    '''
    assert attention_type in ('dot', 'bilinear')
    if attention_type == 'dot':
        assert Q.shape[-1] == K.shape[-1]

    Q_time = Q.shape[1]
    K_time = K.shape[1]

    if attention_type == 'dot':
        logits = op.dot_sim(Q, K)  # [batch, Q_time, time]
    if attention_type == 'bilinear':
        logits = op.bilinear_sim(Q, K)

    if is_mask:
        mask = op.mask(Q_lengths, K_lengths, Q_time, K_time)  # [batch, Q_time, K_time]
        # mask = tf.Print(mask, [logits[0], mask[0]], tf.get_variable_scope().name + " logits, mask: ", summarize=10)
        logits = mask * logits + (1 - mask) * mask_value
        # logits = tf.Print(logits, [logits[0]], tf.get_variable_scope().name + " masked logits: ", summarize=10)

    attention = tf.nn.softmax(logits)

    if drop_prob is not None:
        log.info('use attention drop')
        attention = tf.nn.dropout(attention, drop_prob)

    return op.weighted_sum(attention, V)


def FFN(x, out_dimension_0=None, out_dimension_1=None):
    '''Add two dense connected layer, max(0, x*W0+b0)*W1+b1.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Returns:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    with tf.variable_scope('FFN_1'):
        y = op.dense(x, out_dimension_0, initializer=tf.keras.initializers.he_normal(seed=42))
        y = tf.nn.relu(y)
    with tf.variable_scope('FFN_2'):
        # z = op.dense(y, out_dimension_1, initializer=tf.keras.initializers.glorot_uniform(seed=42))  # TODO: check
        z = op.dense(y, out_dimension_1)  # , add_bias=False)  #!!!!
    return z


def block(
        Q, K, V,
        Q_lengths, K_lengths,
        attention_type='dot',
        is_layer_norm=True,
        is_mask=True, mask_value=-2 ** 32 + 1,
        drop_prob=None):
    '''Add a block unit from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Q: a tensor with shape [batch, Q_time, Q_dimension]
        K: a tensor with shape [batch, time, K_dimension]
        V: a tensor with shape [batch, time, V_dimension]

        Q_length: a tensor with shape [batch]
        K_length: a tensor with shape [batch]

    Returns:
        a tensor with shape [batch, time, dimension]

    Raises:
    '''
    att = attention(Q, K, V,
                    Q_lengths, K_lengths,
                    attention_type=attention_type,
                    is_mask=is_mask, mask_value=mask_value,
                    drop_prob=drop_prob)
    if is_layer_norm:
        with tf.variable_scope('attention_layer_norm'):
            y = op.layer_norm_debug(Q + att)
    else:
        y = Q + att

    z = FFN(y)
    if is_layer_norm:
        with tf.variable_scope('FFN_layer_norm'):
            w = op.layer_norm_debug(y + z)
    else:
        w = y + z
    return w


def CNN(x, out_channels, filter_size, pooling_size, add_relu=True):
    '''Add a convlution layer with relu and max pooling layer.

    Args:
        x: a tensor with shape [batch, in_height, in_width, in_channels]
        out_channels: a number
        filter_size: a number
        pooling_size: a number

    Returns:
        a flattened tensor with shape [batch, num_features]

    Raises:
    '''
    # calculate the last dimension of return
    num_features = ((tf.shape(x)[1] - filter_size + 1) / pooling_size *
                    (tf.shape(x)[2] - filter_size + 1) / pooling_size) * out_channels

    in_channels = x.shape[-1]
    weights = tf.get_variable(
        name='filter',
        shape=[filter_size, filter_size, in_channels, out_channels],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias = tf.get_variable(
        name='bias',
        shape=[out_channels],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding="VALID")
    conv = conv + bias

    if add_relu:
        conv = tf.nn.relu(conv)

    pooling = tf.nn.max_pool(
        conv,
        ksize=[1, pooling_size, pooling_size, 1],
        strides=[1, pooling_size, pooling_size, 1],
        padding="VALID")

    return tf.contrib.layers.flatten(pooling)


def CNN_3d(x, out_channels_0, out_channels_1, add_relu=True):
    '''Add a 3d convlution layer with relu and max pooling layer.

    Args:
        x: a tensor with shape [batch, in_depth, in_height, in_width, in_channels]
        out_channels: a number
        filter_size: a number
        pooling_size: a number

    Returns:
        a flattened tensor with shape [batch, num_features]

    Raises:
    '''
    in_channels = x.shape[-1]
    weights_0 = tf.get_variable(
        name='filter_0',
        shape=[3, 3, 3, in_channels, out_channels_0],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.001, 0.001))
    bias_0 = tf.get_variable(
        name='bias_0',
        shape=[out_channels_0],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_0 = tf.nn.conv3d(x, weights_0, strides=[1, 1, 1, 1, 1], padding="SAME")
    log.info('conv_0 shape: %s' % conv_0.shape)
    conv_0 = conv_0 + bias_0

    if add_relu:
        conv_0 = tf.nn.elu(conv_0)

    pooling_0 = tf.nn.max_pool3d(
        conv_0,
        ksize=[1, 3, 3, 3, 1],
        strides=[1, 3, 3, 3, 1],
        padding="SAME")
    log.info('pooling_0 shape: %s' % pooling_0.shape)

    # layer_1
    weights_1 = tf.get_variable(
        name='filter_1',
        shape=[3, 3, 3, out_channels_0, out_channels_1],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.001, 0.001))
    bias_1 = tf.get_variable(
        name='bias_1',
        shape=[out_channels_1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_1 = tf.nn.conv3d(pooling_0, weights_1, strides=[1, 1, 1, 1, 1], padding="SAME")
    log.info('conv_1 shape: %s' % conv_1.shape)
    conv_1 = conv_1 + bias_1

    if add_relu:
        conv_1 = tf.nn.elu(conv_1)

    pooling_1 = tf.nn.max_pool3d(
        conv_1,
        ksize=[1, 3, 3, 3, 1],
        strides=[1, 3, 3, 3, 1],
        padding="SAME")
    log.info('pooling_1 shape: %s' % pooling_1.shape)

    return tf.contrib.layers.flatten(pooling_1)


def CNN_3d_2d(x, out_channels_0, out_channels_1, add_relu=True):
    '''Add a 3d convlution layer with relu and max pooling layer.

    Args:
        x: a tensor with shape [batch, in_depth, in_height, in_width, in_channels]
        out_channels: a number
        filter_size: a number
        pooling_size: a number

    Returns:
        a flattened tensor with shape [batch, num_features]

    Raises:
    '''
    in_channels = x.shape[-1]
    weights_0 = tf.get_variable(
        name='filter_0',
        shape=[1, 3, 3, in_channels, out_channels_0],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_0 = tf.get_variable(
        name='bias_0',
        shape=[out_channels_0],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_0 = tf.nn.conv3d(x, weights_0, strides=[1, 1, 1, 1, 1], padding="SAME")
    log.info('conv_0 shape: %s' % conv_0.shape)
    conv_0 = conv_0 + bias_0

    if add_relu:
        conv_0 = tf.nn.elu(conv_0)

    pooling_0 = tf.nn.max_pool3d(
        conv_0,
        ksize=[1, 1, 3, 3, 1],
        strides=[1, 1, 3, 3, 1],
        padding="SAME")
    log.info('pooling_0 shape: %s' % pooling_0.shape)

    # layer_1
    weights_1 = tf.get_variable(
        name='filter_1',
        shape=[1, 3, 3, out_channels_0, out_channels_1],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_1 = tf.get_variable(
        name='bias_1',
        shape=[out_channels_1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    conv_1 = tf.nn.conv3d(pooling_0, weights_1, strides=[1, 1, 1, 1, 1], padding="SAME")
    log.info('conv_1 shape: %s' % conv_1.shape)
    conv_1 = conv_1 + bias_1

    if add_relu:
        conv_1 = tf.nn.elu(conv_1)

    pooling_1 = tf.nn.max_pool3d(
        conv_1,
        ksize=[1, 1, 3, 3, 1],
        strides=[1, 1, 3, 3, 1],
        padding="SAME")
    log.info('pooling_1 shape: %s' % pooling_1.shape)

    return tf.contrib.layers.flatten(pooling_1)


def CNN_3d_change(x, out_channels_0, out_channels_1, add_relu=True):
    '''Add a 3d convlution layer with relu and max pooling layer.

    Args:
        x: a tensor with shape [batch, in_depth, in_height, in_width, in_channels]
        out_channels: a number
        filter_size: a number
        pooling_size: a number

    Returns:
        a flattened tensor with shape [batch, num_features]

    Raises:
    '''
    in_channels = x.shape[-1]
    weights_0 = tf.get_variable(
        name='filter_0',
        shape=[3, 3, 3, in_channels, out_channels_0],
        dtype=tf.float32,
        # initializer=tf.random_normal_initializer(0, 0.05))
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    bias_0 = tf.get_variable(
        name='bias_0',
        shape=[out_channels_0],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())
    # Todo
    g_0 = tf.get_variable(name='scale_0',
                          shape=[out_channels_0],
                          dtype=tf.float32,
                          initializer=tf.ones_initializer())
    weights_0 = tf.reshape(g_0, [1, 1, 1, out_channels_0]) * tf.nn.l2_normalize(weights_0, [0, 1, 2])

    conv_0 = tf.nn.conv3d(x, weights_0, strides=[1, 1, 1, 1, 1], padding="VALID")
    log.info('conv_0 shape: %s' % conv_0.shape)
    conv_0 = conv_0 + bias_0
    #######
    '''
    with tf.variable_scope('layer_0'):
        conv_0 = op.layer_norm(conv_0, axis=[1, 2, 3, 4])
        log.info('layer_norm in cnn')
    '''
    if add_relu:
        conv_0 = tf.nn.elu(conv_0)

    pooling_0 = tf.nn.max_pool3d(
        conv_0,
        ksize=[1, 2, 3, 3, 1],
        strides=[1, 2, 3, 3, 1],
        padding="VALID")
    log.info('pooling_0 shape: %s' % pooling_0.shape)

    # layer_1
    weights_1 = tf.get_variable(
        name='filter_1',
        shape=[2, 2, 2, out_channels_0, out_channels_1],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))

    bias_1 = tf.get_variable(
        name='bias_1',
        shape=[out_channels_1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    g_1 = tf.get_variable(name='scale_1',
                          shape=[out_channels_1],
                          dtype=tf.float32,
                          initializer=tf.ones_initializer())
    weights_1 = tf.reshape(g_1, [1, 1, 1, out_channels_1]) * tf.nn.l2_normalize(weights_1, [0, 1, 2])

    conv_1 = tf.nn.conv3d(pooling_0, weights_1, strides=[1, 1, 1, 1, 1], padding="VALID")
    log.info('conv_1 shape: %s' % conv_1.shape)
    conv_1 = conv_1 + bias_1
    # with tf.variable_scope('layer_1'):
    #    conv_1 = op.layer_norm(conv_1, axis=[1, 2, 3, 4])

    if add_relu:
        conv_1 = tf.nn.elu(conv_1)

    pooling_1 = tf.nn.max_pool3d(
        conv_1,
        ksize=[1, 3, 3, 3, 1],
        strides=[1, 3, 3, 3, 1],
        padding="VALID")
    log.info('pooling_1 shape: %s' % pooling_1.shape)

    return tf.contrib.layers.flatten(pooling_1)


def RNN_last_state(x, lengths, hidden_size):
    '''encode x with a gru cell and return the last state.
    
    Args:
        x: a tensor with shape [batch, time, dimension]
        length: a tensor with shape [batch]

    Return:
        a tensor with shape [batch, hidden_size]

    Raises:
    '''
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    outputs, last_states = tf.nn.dynamic_rnn(cell, x, lengths, dtype=tf.float32)
    return outputs, last_states
