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

import math
from logging import getLogger

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

log = getLogger(__name__)


def learning_rate(step_num, d_model=512, warmup_steps=4000):
    a = step_num ** (-0.5)
    b = step_num * warmup_steps ** (-1.5)
    return a, b, d_model ** (-0.5) * min(step_num ** (-0.5), step_num * (warmup_steps ** (-1.5)))


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    log.info('use selu')
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def bilinear_sim_4d(x, y, is_nor=True):
    '''calulate bilinear similarity with two 4d tensor.
    
    Args:
        x: a tensor with shape [batch, time_x, dimension_x, num_stacks]
        y: a tensor with shape [batch, time_y, dimension_y, num_stacks]

    Returns:
        a tensor with shape [batch, time_x, time_y, num_stacks]

    Raises:
        ValueError: if
            the shapes of x and y are not match;
            bilinear matrix reuse error.
    '''
    M = tf.get_variable(
        name="bilinear_matrix",
        shape=[x.shape[2], y.shape[2], x.shape[3]],
        dtype=tf.float32,
        initializer=tf.orthogonal_initializer())
    sim = tf.einsum('biks,kls,bjls->bijs', x, M, y)

    if is_nor:
        scale = tf.sqrt(tf.cast(x.shape[2] * y.shape[2], tf.float32))
        scale = tf.maximum(1.0, scale)
        return sim / scale
    else:
        return sim


def bilinear_sim(x, y, is_nor=True):
    '''calculate bilinear similarity with two tensor.
    Args:
        x: a tensor with shape [batch, time_x, dimension_x]
        y: a tensor with shape [batch, time_y, dimension_y]
    
    Returns:
        a tensor with shape [batch, time_x, time_y]
    Raises:
        ValueError: if
            the shapes of x and y are not match;
            bilinear matrix reuse error.
    '''
    M = tf.get_variable(
        name="bilinear_matrix",
        shape=[x.shape[-1], y.shape[-1]],
        dtype=tf.float32,
        # initializer=tf.orthogonal_initializer())
        initializer=tf.keras.initializers.glorot_uniform(seed=42))
    sim = tf.einsum('bik,kl,bjl->bij', x, M, y)

    if is_nor:
        scale = tf.sqrt(tf.cast(x.shape[-1] * y.shape[-1], tf.float32))
        scale = tf.maximum(1.0, scale)
        return sim / scale
    else:
        return sim


def dot_sim(x, y, is_nor=True):
    '''calculate dot similarity with two tensor.

    Args:
        x: a tensor with shape [batch, time_x, dimension]
        y: a tensor with shape [batch, time_y, dimension]
    
    Returns:
        a tensor with shape [batch, time_x, time_y]
    Raises:
        AssertionError: if
            the shapes of x and y are not match.
    '''
    assert x.shape[-1] == y.shape[-1]

    sim = tf.einsum('bik,bjk->bij', x, y)

    if is_nor:
        scale = tf.sqrt(tf.cast(x.shape[-1], tf.float32))
        scale = tf.maximum(1.0, scale)
        return sim / scale
    else:
        return sim


def layer_norm(x, axis=None, epsilon=1e-6):
    '''Add layer normalization.

    Args:
        x: a tensor
        axis: the dimensions to normalize

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    log.info('wrong version of layer_norm')
    scale = tf.get_variable(
        name='scale',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    bias = tf.get_variable(
        name='bias',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    if axis is None:
        axis = [-1]

    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=axis, keepdims=True)
    norm = (x - mean) * tf.rsqrt(variance + epsilon)
    return scale * norm + bias


def layer_norm_debug(x, axis=None, epsilon=1e-6):
    '''Add layer normalization.

    Args:
        x: a tensor
        axis: the dimensions to normalize

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    if axis is None:
        axis = [-1]
    shape = [x.shape[i] for i in axis]

    scale = tf.get_variable(
        name='scale',
        shape=shape,
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    bias = tf.get_variable(
        name='bias',
        shape=shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer())

    mean = tf.reduce_mean(x, axis=axis, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=axis, keepdims=True)
    norm = (x - mean) * tf.rsqrt(variance + epsilon)
    return scale * norm + bias


def dense(x, out_dimension=None, add_bias=True, initializer=tf.orthogonal_initializer()):
    '''Add dense connected layer, Wx + b.

    Args:
        x: a tensor with shape [batch, time, dimension]
        out_dimension: a number which is the output dimension

    Return:
        a tensor with shape [batch, time, out_dimension]

    Raises:
    '''
    if out_dimension is None:
        out_dimension = x.shape[-1]

    W = tf.get_variable(
        name='weights',
        shape=[x.shape[-1], out_dimension],
        dtype=tf.float32,
        initializer=initializer)
    if add_bias:
        bias = tf.get_variable(
            name='bias',
            shape=[1],
            dtype=tf.float32,
            initializer=tf.zeros_initializer())
        return tf.einsum('bik,kj->bij', x, W) + bias
    else:
        return tf.einsum('bik,kj->bij', x, W)


def matmul_2d(x, out_dimension, drop_prob=None):
    '''Multiplies 2-d tensor by weights.

    Args:
        x: a tensor with shape [batch, dimension]
        out_dimension: a number

    Returns:
        a tensor with shape [batch, out_dimension]

    Raises:
    '''
    W = tf.get_variable(
        name='weights',
        shape=[x.shape[1], out_dimension],
        dtype=tf.float32,
        initializer=tf.orthogonal_initializer())
    if drop_prob is not None:
        W = tf.nn.dropout(W, drop_prob)
        log.info('W is dropout')

    return tf.matmul(x, W)


def gauss_positional_encoding_vector(x, role=0, value=0):
    position = int(x.shape[1])
    dimension = int(x.shape[2])
    log.info('position: %s' % position)
    log.info('dimension: %s' % dimension)

    _lambda = tf.get_variable(
        name='lambda',
        shape=[position],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value))
    _lambda = tf.expand_dims(_lambda, axis=-1)

    mean = [position / 2.0, dimension / 2.0]

    # cov = [[position/3.0, 0], [0, dimension/3.0]]
    sigma_x = position / math.sqrt(4.0 * dimension)
    sigma_y = math.sqrt(dimension / 4.0)
    cov = [[sigma_x * sigma_x, role * sigma_x * sigma_y],
           [role * sigma_x * sigma_y, sigma_y * sigma_y]]

    pos = np.dstack(np.mgrid[0:position, 0:dimension])

    rv = multivariate_normal(mean, cov)
    signal = rv.pdf(pos)
    signal = signal - np.max(signal) / 2.0

    signal = tf.multiply(_lambda, signal)
    signal = tf.expand_dims(signal, axis=0)

    log.info('gauss positional encoding')

    return x + _lambda * signal


def positional_encoding(x, min_timescale=1.0, max_timescale=1.0e4, value=0):
    '''Adds a bunch of sinusoids of different frequencies to a tensor.

    Args:
        x: a tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    length = x.shape[1]
    channels = x.shape[2]
    _lambda = tf.get_variable(
        name='lambda',
        shape=[1],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value))

    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    # signal = tf.reshape(signal, [1, length, channels])
    signal = tf.expand_dims(signal, axis=0)

    return x + _lambda * signal


def positional_encoding_vector(x, min_timescale=1.0, max_timescale=1.0e4, value=0):
    '''Adds a bunch of sinusoids of different frequencies to a tensor.

    Args:
        x: a tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a tensor the same shape as x.

    Raises:
    '''
    length = x.shape[1]
    channels = x.shape[2]
    _lambda = tf.get_variable(
        name='lambda',
        shape=[length],
        dtype=tf.float32,
        initializer=tf.constant_initializer(value))
    _lambda = tf.expand_dims(_lambda, axis=-1)

    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])

    signal = tf.multiply(_lambda, signal)
    signal = tf.expand_dims(signal, axis=0)

    return x + signal


def mask(row_lengths, col_lengths, max_row_length, max_col_length):
    '''Return a mask tensor representing the first N positions of each row and each column.

    Args:
        row_lengths: a tensor with shape [batch]
        col_lengths: a tensor with shape [batch]

    Returns:
        a mask tensor with shape [batch, max_row_length, max_col_length]

    Raises:
    '''
    row_mask = tf.sequence_mask(row_lengths, max_row_length)  # bool, [batch, max_row_len]
    col_mask = tf.sequence_mask(col_lengths, max_col_length)  # bool, [batch, max_col_len]

    row_mask = tf.cast(tf.expand_dims(row_mask, -1), tf.float32)
    col_mask = tf.cast(tf.expand_dims(col_mask, -1), tf.float32)

    return tf.einsum('bik,bjk->bij', row_mask, col_mask)


def weighted_sum(weight, values):
    '''Calcualte the weighted sum.

    Args:
        weight: a tensor with shape [batch, time, dimension]
        values: a tensor with shape [batch, dimension, values_dimension]

    Return:
        a tensor with shape [batch, time, values_dimension]

    Raises:
    '''
    return tf.einsum('bij,bjk->bik', weight, values)
