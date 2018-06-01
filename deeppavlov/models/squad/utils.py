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


class PtrNet:
    def __init__(self, cell_size, keep_prob=1.0, scope="ptr_net"):
        self.gru = tf.nn.rnn_cell.GRUCell(cell_size)
        self.scope = scope
        self.keep_prob = keep_prob

    def __call__(self, init, match, hidden_size, mask):
        with tf.variable_scope(self.scope):
            BS, ML, MH = tf.unstack(tf.shape(match))
            BS, IH = tf.unstack(tf.shape(init))
            match_do = tf.nn.dropout(match, keep_prob=self.keep_prob, noise_shape=[BS, 1, MH])
            dropout_mask = tf.nn.dropout(tf.ones([BS, IH], dtype=tf.float32), keep_prob=self.keep_prob)
            inp, logits1 = attention(match_do, init * dropout_mask, hidden_size, mask)
            inp_do = tf.nn.dropout(inp, keep_prob=self.keep_prob)
            _, state = self.gru(inp_do, init)
            tf.get_variable_scope().reuse_variables()
            _, logits2 = attention(match_do, state * dropout_mask, hidden_size, mask)
            return logits1, logits2


def dot_attention(inputs, memory, mask, att_size, keep_prob=1.0, scope="dot_attention"):
    """Computes attention vector for each item in inputs:
       attention vector is a weighted sum of memory items.
       Dot product between input and memory vector is used as similarity measure.

       Gate mechanism is applied to attention vectors to produce output.

    Args:
        inputs: Tensor [batch_size x input_len x feature_size]
        memory: Tensor [batch_size x memory_len x feature_size]
        mask: inputs mask
        att_size: hidden size of attention
        keep_prob: dropout keep_prob
        scope:

    Returns:
        attention vectors [batch_size x input_len x (feature_size + feature_size)]

    """
    with tf.variable_scope(scope):
        BS, IL, IH = tf.unstack(tf.shape(inputs))
        BS, ML, MH = tf.unstack(tf.shape(memory))

        d_inputs = tf.nn.dropout(inputs, keep_prob=keep_prob, noise_shape=[BS, 1, IH])
        d_memory = tf.nn.dropout(memory, keep_prob=keep_prob, noise_shape=[BS, 1, MH])

        with tf.variable_scope("attention"):
            inputs_att = tf.layers.dense(d_inputs, att_size, use_bias=False, activation=tf.nn.relu)
            memory_att = tf.layers.dense(d_memory, att_size, use_bias=False, activation=tf.nn.relu)
            logits = tf.matmul(inputs_att, tf.transpose(memory_att, [0, 2, 1])) / (att_size ** 0.5)
            mask = tf.tile(tf.expand_dims(mask, axis=1), [1, IL, 1])
            att_weights = tf.nn.softmax(softmax_mask(logits, mask))
            outputs = tf.matmul(att_weights, memory)
            res = tf.concat([inputs, outputs], axis=2)

        with tf.variable_scope("gate"):
            dim = res.get_shape().as_list()[-1]
            d_res = tf.nn.dropout(res, keep_prob=keep_prob, noise_shape=[BS, 1, IH + MH])
            gate = tf.layers.dense(d_res, dim, use_bias=False, activation=tf.nn.sigmoid)
            return res * gate


def simple_attention(memory, att_size, mask, keep_prob=1.0, scope="simple_attention"):
    """Simple attention without any conditions.

       Computes weighted sum of memory elements.
    """
    with tf.variable_scope(scope):
        BS, ML, MH = tf.unstack(tf.shape(memory))
        memory_do = tf.nn.dropout(memory, keep_prob=keep_prob, noise_shape=[BS, 1, MH])
        logits = tf.layers.dense(tf.layers.dense(memory_do, att_size, activation=tf.nn.tanh), 1, use_bias=False)
        logits = softmax_mask(tf.squeeze(logits, [2]), mask)
        att_weights = tf.expand_dims(tf.nn.softmax(logits), axis=2)
        res = tf.reduce_sum(att_weights * memory, axis=1)
        return res


def attention(inputs, state, att_size, mask, scope="attention"):
    """Computes weighted sum of inputs conditioned on state"""
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        logits = tf.layers.dense(tf.layers.dense(u, att_size, activation=tf.nn.tanh), 1, use_bias=False)
        logits = softmax_mask(tf.squeeze(logits, [2]), mask)
        att_weights = tf.expand_dims(tf.nn.softmax(logits), axis=2)
        res = tf.reduce_sum(att_weights * inputs, axis=1)
        return res, logits


def softmax_mask(val, mask):
    INF = 1e30
    return -INF * (1 - tf.cast(mask, tf.float32)) + val
