# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Reshape, Concatenate, Lambda, Embedding, Conv2D, Activation, Input
from keras.engine.topology import Layer
from keras.layers.merge import Multiply, Add
from keras.activations import softmax
import numpy as np
import tensorflow as tf


def expand_tile(units, axis):
    """
    Expand and tile tensor along given axis

    Args:
        units: tf tensor with dimensions [batch_size, time_steps, n_input_features]
        axis: axis along which expand and tile. Must be 1 or 2

    """
    assert axis in (1, 2)
    n_time_steps = K.int_shape(units)[1]
    repetitions = [1, 1, 1, 1]
    repetitions[axis] = n_time_steps
    if axis == 1:
        expanded = Reshape(target_shape=( (1,) + K.int_shape(units)[1:] ))(units)
    else:
        expanded = Reshape(target_shape=(K.int_shape(units)[1:2] + (1,) + K.int_shape(units)[2:]))(units)
    return K.tile(expanded, repetitions)


def softvaxaxis2(x):
    return softmax(x, axis=2)


def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """
    Compute additive self attention for time series of vectors (with batch dimension)
            the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
            v is a learnable vector of n_hidden dimensionality,
            W_1 and W_2 are learnable [n_hidden, n_input_features] matrices

    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of2784131 units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output

    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
        """
    n_input_features = K.int_shape(units)[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
    exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
    units_pairs = Concatenate(axis=3)([exp1, exp2])
    query = Dense(n_hidden, activation="tanh")(units_pairs)
    attention = Dense(1, activation=softvaxaxis2)(query)
    attended_units = Lambda(lambda x: K.sum(attention * x, axis=2))(exp1)
    output = Dense(n_output_features, activation=activation)(attended_units)
    return output


def multiplicative_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """
    Compute multiplicative self attention for time series of vectors (with batch dimension)
    the formula: score(h_i, h_j) = <W_1 h_i,  W_2 h_j>,  W_1 and W_2 are learnable matrices
    with dimensionality [n_hidden, n_input_features]

    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output

    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
    """
    n_input_features = K.int_shape(units)[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
    exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
    queries = Dense(n_hidden)(exp1)
    keys = Dense(n_hidden)(exp2)
    scores = Lambda(lambda x: K.sum(queries * x, axis=3, keepdims=True))(keys)
    attention = Lambda(lambda x: softvaxaxis2(x))(scores)
    mult = Multiply()([attention, exp1])
    attended_units = Lambda(lambda x: K.sum(x, axis=2))(mult)
    output = Dense(n_output_features, activation=activation)(attended_units)
    return output


def multiplicative_self_attention_init(n_hidden, n_output_features, activation):
    layers = {}
    layers["queries"] = Dense(n_hidden)
    layers["keys"] = Dense(n_hidden)
    layers["output"] = Dense(n_output_features, activation=activation)
    return layers


def multiplicative_self_attention_get_output(units, layers):
    exp1 = Lambda(lambda x: expand_tile(x, axis=1))(units)
    exp2 = Lambda(lambda x: expand_tile(x, axis=2))(units)
    queries = layers["queries"](exp1)
    keys = layers["keys"](exp2)
    scores = Lambda(lambda x: K.sum(queries * x, axis=3, keepdims=True))(keys)
    attention = Lambda(lambda x: softvaxaxis2(x))(scores)
    mult = Multiply()([attention, exp1])
    attended_units = Lambda(lambda x: K.sum(x, axis=2))(mult)
    output = layers["output"](attended_units)
    return output


def char_emb_cnn_func(n_characters: int,
                      char_embedding_dim: int,
                      emb_mat: np.array = None,
                      filter_widths=(3, 4, 5, 7),
                      highway_on_top=False):

    emb_layer = Embedding(n_characters,
                          char_embedding_dim)

    if emb_mat is not None:
        emb_layer.set_weights([emb_mat])

    conv2d_layers = []
    for filter_width in filter_widths:
        conv2d_layers.append(Conv2D(char_embedding_dim,
                                    (1, filter_width),
                                    padding='same'))

    if highway_on_top:
        dense1 = Dense(char_embedding_dim * len(filter_widths))
        dense2 = Dense(char_embedding_dim * len(filter_widths))

    def result(input):
        emb_c = emb_layer(input)
        conv_results_list = []
        for cl in conv2d_layers:
            conv_results_list.append(cl(emb_c))
        emb_c = Lambda(lambda x: K.concatenate(x, axis=3))(conv_results_list)
        emb_c = Lambda(lambda x: K.max(x, axis=2))(emb_c)
        if highway_on_top:
            sigmoid_gate = dense1(emb_c)
            sigmoid_gate = Activation('sigmoid')(sigmoid_gate)
            deeper_units = dense2(emb_c)
            emb_c = Add()([Multiply()([sigmoid_gate, deeper_units]),
                           Multiply()([Lambda(lambda x: K.constant(1., shape=K.shape(x)) - x)(sigmoid_gate), emb_c])])
            emb_c = Activation('relu')(emb_c)
        return emb_c

    return result


def masking_sequences(sequences, seq_lengths):
    """
    Function extracts seq_lengths[i] element for each sequences[i].
    Useful for extracting corresponding hidden state of RNN output.

    Args:
        sequences: tensor of size (batch_size, timesteps, dim)
        seq_lengths: tensor of integers of size (batch_size, 2).
            Each row is a pair (i, length) where length is a number of element of sequences[i] to extract

    Returns:
        tensor of shape (batch_size, dim)
    """
    return Lambda(lambda x: tf.gather_nd(x[0], K.cast(x[1], "int32")))([sequences, seq_lengths])
