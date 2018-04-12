"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

from deeppavlov.skills.go_bot import csoftmax_attention
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


def general_attention(key, context, hidden_size, projected_align=False):
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden_size]
    projected_key = \
        tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    r_projected_key = tf.reshape(projected_key, shape=[-1, hidden_size, 1])

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=r_context,
                                        dtype=tf.float32)
    # bilstm_output: [-1, max_num_tokens, hidden_size]
    bilstm_output = tf.concat([output_fw, output_bw], -1)

    attn = tf.nn.softmax(tf.matmul(bilstm_output, r_projected_key), dim=1)

    if projected_align:
        log.info("Using projected attention alignment")
        t_context = tf.transpose(bilstm_output, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, hidden_size])
    else:
        log.info("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])
    return output


def light_general_attention(key, context, hidden_size, projected_align=False):
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden_size]
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    r_projected_key = tf.reshape(projected_key, shape=[-1, hidden_size, 1])

    # projected context: [None, None, hidden_size]
    projected_context = \
        tf.layers.dense(r_context, hidden_size, kernel_initializer=xav())

    attn = tf.nn.softmax(tf.matmul(projected_context, r_projected_key), dim=1)

    if projected_align:
        log.info("Using projected attention alignment")
        t_context = tf.transpose(projected_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, hidden_size])
    else:
        log.info("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])
    return output


def cs_general_attention(key, context, hidden_size, depth, projected_align=False):
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")
    key_size = tf.shape(key)[-1]
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])
    # projected_context: [None, max_num_tokens, token_size]
    projected_context = tf.layers.dense(r_context, token_size,
                                        kernel_initializer=xav(),
                                        name='projected_context')

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=projected_context,
                                        dtype=tf.float32)
    # bilstm_output: [-1, max_num_tokens, hidden_size]
    bilstm_output = tf.concat([output_fw, output_bw], -1)
    h_state_for_sketch = bilstm_output

    if projected_align:
        log.info("Using projected attention alignment")
        h_state_for_attn_alignment = bilstm_output
        aligned_h_state = csoftmax_attention.attention_gen_block(
            h_state_for_sketch, h_state_for_attn_alignment, key, depth)
        output = \
            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * hidden_size])
    else:
        log.info("Using without projected attention alignment")
        h_state_for_attn_alignment = projected_context
        aligned_h_state = csoftmax_attention.attention_gen_block(
            h_state_for_sketch, h_state_for_attn_alignment, key, depth)
        output = \
            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * token_size])
    return output


def bahdanau_attention(key, context, hidden_size, projected_align=False):
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden_size]
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    r_projected_key = \
        tf.tile(tf.reshape(projected_key, shape=[-1, 1, hidden_size]),
                [1, max_num_tokens, 1])

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=r_context,
                                        dtype=tf.float32)

    # bilstm_output: [-1,self.max_num_tokens,_n_hidden]
    bilstm_output = tf.concat([output_fw, output_bw], -1)
    concat_h_state = tf.concat([r_projected_key, output_fw, output_bw], -1)
    projected_state = \
        tf.layers.dense(concat_h_state, hidden_size, use_bias=False,
                        kernel_initializer=xav())
    score = \
        tf.layers.dense(tf.tanh(projected_state), units=1, use_bias=False,
                        kernel_initializer=xav())

    attn = tf.nn.softmax(score, dim=1)

    if projected_align:
        log.info("Using projected attention alignment")
        t_context = tf.transpose(bilstm_output, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, hidden_size])
    else:
        log.info("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])
    return output


def light_bahdanau_attention(key, context, hidden_size, projected_align=False):
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]
    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])

    # projected_key: [None, None, hidden_size]
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav())
    r_projected_key = \
        tf.tile(tf.reshape(projected_key, shape=[-1, 1, hidden_size]),
                [1, max_num_tokens, 1])

    # projected_context: [None, max_num_tokens, hidden_size]
    projected_context = \
        tf.layers.dense(r_context, hidden_size, kernel_initializer=xav())
    concat_h_state = tf.concat([projected_context, r_projected_key], -1)

    projected_state = \
        tf.layers.dense(concat_h_state, hidden_size, use_bias=False,
                        kernel_initializer=xav())
    score = \
        tf.layers.dense(tf.tanh(projected_state), units=1, use_bias=False,
                        kernel_initializer=xav())

    attn = tf.nn.softmax(score, dim=1)

    if projected_align:
        log.info("Using projected attention alignment")
        t_context = tf.transpose(projected_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, hidden_size])
    else:
        log.info("Using without projected attention alignment")
        t_context = tf.transpose(r_context, [0, 2, 1])
        output = tf.reshape(tf.matmul(t_context, attn),
                            shape=[batch_size, -1, token_size])
    return output


def cs_bahdanau_attention(key, context, hidden_size, depth, projected_align=False):
    if hidden_size % 2 != 0:
        raise ValueError("hidden size must be dividable by two")
    batch_size = tf.shape(context)[0]
    max_num_tokens, token_size = context.get_shape().as_list()[-2:]

    r_context = tf.reshape(context, shape=[-1, max_num_tokens, token_size])
    # projected context: [None, max_num_tokens, token_size]
    projected_context = tf.layers.dense(r_context, token_size,
                                        kernel_initializer=xav(),
                                        name='projected_context')

    # projected_key: [None, None, hidden_size]
    projected_key = tf.layers.dense(key, hidden_size, kernel_initializer=xav(),
                                    name='projected_key')
    r_projected_key = \
        tf.tile(tf.reshape(projected_key, shape=[-1, 1, hidden_size]),
                [1, max_num_tokens, 1])

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_size//2)
    (output_fw, output_bw), states = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                        cell_bw=lstm_bw_cell,
                                        inputs=projected_context,
                                        dtype=tf.float32)

    # bilstm_output: [-1, max_num_tokens, hidden_size]
    bilstm_output = tf.concat([output_fw, output_bw], -1)
    concat_h_state = tf.concat([r_projected_key, output_fw, output_bw], -1)

    if projected_align:
        log.info("Using projected attention alignment")
        h_state_for_attn_alignment = bilstm_output
        aligned_h_state = csoftmax_attention.attention_bah_block(
            concat_h_state, h_state_for_attn_alignment, depth)
        output = \
            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * hidden_size])
    else:
        log.info("Using without projected attention alignment")
        h_state_for_attn_alignment = projected_context
        aligned_h_state = csoftmax_attention.attention_bah_block(
            concat_h_state, h_state_for_attn_alignment, depth)
        output = \
            tf.reshape(aligned_h_state, shape=[batch_size, -1, depth * token_size])
    return output

