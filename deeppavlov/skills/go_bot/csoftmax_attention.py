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
import numpy as np
import tensorflow as tf
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


def csoftmax_for_slice(input):
    """
    Compute the constrained softmax (csoftmax);
    See paper "Learning What's Easy: Fully Differentiable Neural Easy-First Taggers"
    on https://andre-martins.github.io/docs/emnlp2017_final.pdf (page 4)
    :param input: [input tensor, cumulative attention]
    :return: distribution
    """

    [ten, u] = input

    shape_t = ten.shape
    shape_u = u.shape

    ten -= tf.reduce_mean(ten)
    q = tf.exp(ten)
    active = tf.ones_like(u, dtype=tf.int32)
    mass = tf.constant(0, dtype=tf.float32)
    found = tf.constant(True, dtype=tf.bool)

    def loop(q_, mask, mass_, found_):
        q_list = tf.dynamic_partition(q_, mask, 2)
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(q_)[0]), mask, 2)  # 0 element it False,
        #  1 element if true

        p = q_list[1] * (1.0 - mass_) / tf.reduce_sum(q_list[1])
        p_new = tf.dynamic_stitch(condition_indices, [q_list[0], p])

        # verification of the condition and modification of masks
        less_mask = tf.cast(tf.less(u, p_new), tf.int32)  # 0 when u bigger than p, 1 when u less than p
        condition_indices = tf.dynamic_partition(tf.range(tf.shape(p_new)[0]), less_mask,
                                                 2)  # 0 when u bigger
        #  than p, 1 when u less than p

        split_p_new = tf.dynamic_partition(p_new, less_mask, 2)
        split_u = tf.dynamic_partition(u, less_mask, 2)

        alpha = tf.dynamic_stitch(condition_indices, [split_p_new[0], split_u[1]])
        mass_ += tf.reduce_sum(split_u[1])

        mask = mask * (tf.ones_like(less_mask) - less_mask)

        found_ = tf.cond(tf.equal(tf.reduce_sum(less_mask), 0),
                         lambda: False,
                         lambda: True)

        alpha = tf.reshape(alpha, q_.shape)

        return alpha, mask, mass_, found_

    (csoft, mask_, _, _) = tf.while_loop(cond=lambda _0, _1, _2, f: f,
                                         body=loop,
                                         loop_vars=(q, active, mass, found))

    return [csoft, mask_]

def csoftmax(tensor, inv_cumulative_att):
    shape_ten = tensor.shape
    shape_cum = inv_cumulative_att.shape

    merge_tensor = [tensor, inv_cumulative_att]
    cs, _ = tf.map_fn(csoftmax_for_slice, merge_tensor, dtype=[tf.float32, tf.float32])  # [bs, L]
    return cs

# def attention_step(encoder_outputs, sketch, key, cum_att):
#     with tf.name_scope('attention_step'):
#         encoder_dims = encoder_outputs.get_shape().as_list()
#         batch_size = encoder_dims[0]
#         num_tokens = encoder_dims[1]
#         hidden_size = encoder_dims[2]
#         repeated_sketch = tf.tile(tf.reshape(sketch, [-1, 1, hidden_size]), (1,num_tokens, 1))
#         concat_mem = tf.concat([encoder_outputs, repeated_sketch],-1)
# #         tf.layers.conv1d(before_att, dim_hlayer, window_size * 2 + 1, padding='same')
#
#         concat_mem = tf.reshape(concat_mem, [-1, num_tokens, 2*hidden_size]) # dirty trick
#
#         projected_mem = tf.layers.dense(concat_mem, hidden_size)
#         projected_key = tf.layers.dense(key, hidden_size)
#         t_key = tf.reshape(projected_key,[-1, hidden_size, 1])
#         score = tf.matmul(projected_mem,t_key)
#         score = tf.reshape(tf.matmul(projected_mem,t_key), [-1, num_tokens])
#         inv_cum_att = tf.reshape(tf.ones_like(cum_att) - cum_att, [-1, num_tokens])
#         att = csoftmax(score, inv_cum_att)
#         t_projected_mem = tf.transpose(projected_mem, [0,2,1])
#         t_projected_mem = tf.transpose(projected_mem, [0,2,1])
#         r_att = tf.reshape(att, [-1, num_tokens, 1])
#         next_sketch = tf.squeeze(tf.matmul(t_projected_mem,r_att),-1)
#     return next_sketch, att

# def attention_block(encoder_outputs, key, attention_depth):
#     with tf.name_scope('attention_block'):
#         encoder_dims = tf.shape(encoder_outputs)
#         batch_size = encoder_dims[0]
#         num_tokens = encoder_dims[1]
#         hidden_size = encoder_dims[2]
#
#         sketches = [tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)]
#         cum_att = tf.zeros(shape=[batch_size, num_tokens])  # cumulative attention
#         for i in range(attention_depth):
#             sketch, cum_att_ = attention_step(encoder_outputs, sketches[-1], key, cum_att)
#             sketches.append(sketch) #sketch
#             cum_att += cum_att_
#         final_sketch = tf.reshape(tf.transpose(tf.stack(sketches[1:]), [1, 0, 2]),[batch_size, attention_depth, hidden_size])
#     return final_sketch


def attention_gen_step(hidden_for_sketch, hidden_for_attn_alignment, sketch, key, cum_att):
    with tf.name_scope('attention_step'):
        sketch_dims = hidden_for_sketch.get_shape().as_list()
        batch_size = sketch_dims[0]
        num_tokens = sketch_dims[1]
        hidden_size = sketch_dims[2]
        attn_alignment_dims = hidden_for_attn_alignment.get_shape().as_list()
        attn_alignment_hidden_size = attn_alignment_dims[2]

        repeated_sketch = tf.tile(tf.reshape(sketch, [-1, 1, hidden_size]), (1,num_tokens, 1))
        concat_mem = tf.concat([hidden_for_sketch, repeated_sketch],-1)


        concat_mem = tf.reshape(concat_mem, [-1, num_tokens, 2*hidden_size]) # dirty trick
        reduce_mem = tf.layers.dense(concat_mem, hidden_size)

        projected_key = tf.layers.dense(key, hidden_size)
        t_key = tf.reshape(projected_key,[-1, hidden_size, 1])

        score = tf.reshape(tf.matmul(reduce_mem, t_key), [-1, num_tokens])

        # score = tf.squeeze(tf.layers.dense(reduce_mem, units = 1,
        #                             use_bias=False),-1)
        inv_cum_att = tf.reshape(tf.ones_like(cum_att) - cum_att, [-1, num_tokens])
        att = csoftmax(score, inv_cum_att)

        t_reduce_mem = tf.transpose(reduce_mem, [0,2,1])
        t_hidden_for_attn_alignment = tf.transpose(hidden_for_attn_alignment, [0,2,1])

        r_att = tf.reshape(att, [-1, num_tokens, 1])

        next_sketch = tf.squeeze(tf.matmul(t_reduce_mem,r_att),-1)
        aligned_hidden_sketch = tf.squeeze(tf.matmul(t_hidden_for_attn_alignment,r_att),-1)
    return next_sketch, att, aligned_hidden_sketch

def attention_gen_block(hidden_for_sketch, hidden_for_attn_alignment, key, attention_depth):
    with tf.name_scope('attention_block'):
        sketch_dims = tf.shape(hidden_for_sketch)
        batch_size = sketch_dims[0]
        num_tokens = sketch_dims[1]
        hidden_size = sketch_dims[2]

        attn_alignment_dims = tf.shape(hidden_for_attn_alignment)
        attn_alignment_hidden_size = attn_alignment_dims[2]

        sketches = [tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)]
        aligned_hiddens = []
        cum_att = tf.zeros(shape=[batch_size, num_tokens])  # cumulative attention
        for i in range(attention_depth):
            sketch, cum_att_, aligned_hidden = attention_gen_step(hidden_for_sketch, hidden_for_attn_alignment, sketches[-1], key, cum_att)
            sketches.append(sketch) #sketch
            aligned_hiddens.append(aligned_hidden) #sketch
            cum_att += cum_att_
        final_aligned_hiddens = tf.reshape(tf.transpose(tf.stack(aligned_hiddens), [1, 0, 2]),[1, attention_depth, attn_alignment_hidden_size])
    return final_aligned_hiddens


def attention_bah_step(hidden_for_sketch, hidden_for_attn_alignment, sketch, cum_att):
    with tf.name_scope('attention_step'):
        sketch_dims = hidden_for_sketch.get_shape().as_list()
        batch_size = sketch_dims[0]
        num_tokens = sketch_dims[1]
        hidden_size = sketch_dims[2]
        attn_alignment_dims = hidden_for_attn_alignment.get_shape().as_list()
        attn_alignment_hidden_size = attn_alignment_dims[2]

        repeated_sketch = tf.tile(tf.reshape(sketch, [-1, 1, hidden_size]), (1,num_tokens, 1))
        concat_mem = tf.concat([hidden_for_sketch, repeated_sketch],-1)


        concat_mem = tf.reshape(concat_mem, [-1, num_tokens, 2*hidden_size]) # dirty trick
        reduce_mem = tf.layers.dense(concat_mem, hidden_size)

        score = tf.squeeze(tf.layers.dense(reduce_mem, units = 1,
                                    use_bias=False),-1)
        inv_cum_att = tf.reshape(tf.ones_like(cum_att) - cum_att, [-1, num_tokens])
        att = csoftmax(score, inv_cum_att)

        t_reduce_mem = tf.transpose(reduce_mem, [0,2,1])
        t_hidden_for_attn_alignment = tf.transpose(hidden_for_attn_alignment, [0,2,1])

        r_att = tf.reshape(att, [-1, num_tokens, 1])

        next_sketch = tf.squeeze(tf.matmul(t_reduce_mem,r_att),-1)
        aligned_hidden_sketch = tf.squeeze(tf.matmul(t_hidden_for_attn_alignment,r_att),-1)
    return next_sketch, att, aligned_hidden_sketch

def attention_bah_block(hidden_for_sketch, hidden_for_attn_alignment, attention_depth):
    with tf.name_scope('attention_block'):
        sketch_dims = tf.shape(hidden_for_sketch)
        batch_size = sketch_dims[0]
        num_tokens = sketch_dims[1]
        hidden_size = sketch_dims[2]

        attn_alignment_dims = tf.shape(hidden_for_attn_alignment)
        attn_alignment_hidden_size = attn_alignment_dims[2]

        sketches = [tf.zeros(shape=[batch_size, hidden_size], dtype=tf.float32)]
        aligned_hiddens = []
        cum_att = tf.zeros(shape=[batch_size, num_tokens])  # cumulative attention
        for i in range(attention_depth):
            sketch, cum_att_, aligned_hidden = attention_bah_step(hidden_for_sketch, hidden_for_attn_alignment, sketches[-1], cum_att)
            sketches.append(sketch) #sketch
            aligned_hiddens.append(aligned_hidden) #sketch
            cum_att += cum_att_
        final_aligned_hiddens = tf.reshape(tf.transpose(tf.stack(aligned_hiddens), [1, 0, 2]),[1, attention_depth, attn_alignment_hidden_size])
    return final_aligned_hiddens
