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
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Reshape, Concatenate, Lambda, Embedding, Conv2D, Activation
from keras.engine.topology import Layer
from keras.layers.merge import Multiply
from keras.activations import softmax
import numpy as np

def expand_tile(units, axis):
    """Expand and tile tensor along given axis
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
    """ Computes additive self attention for time series of vectors (with batch dimension)
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
    """ Computes multiplicative self attention for time series of vectors (with batch dimension)
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


def character_embedding_network(char_input,
                                n_characters: int = None,
                                emb_mat: np.array = None,
                                char_embedding_dim: int = None,
                                filter_widths=(3, 4, 5, 7),
                                highway_on_top=False):
    """ Characters to vector. Every sequence of characters (token)
        is embedded to vector space with dimensionality char_embedding_dim
        Convolution plus max_pooling is used to obtain vector representations
        of words.

    Args:
        char_input: numpy array of int32 type with dimensionality [B, T, C]
            B - batch size (can be None)
            T - Number of tokens (can be None)
            C - number of characters (can be None)
        n_characters: total number of unique characters
        emb_mat: if n_characters is not provided the emb_mat should be provided
            it is a numpy array with dimensions [V, E], where V - vocabulary size
            and E - embeddings dimension
        char_embedding_dim: dimensionality of characters embeddings
        filter_widths: array of width of kernel in convolutional embedding network
            used in parallel

    Returns:
        embeddings: tensor with dimensionality [B, T, F],
            where F is dimensionality of embeddings
    """
    assert(emb_mat is not None or n_characters is not None)

    char_emb_var = Embedding(n_characters,
                                 char_embedding_dim)
    if emb_mat is not None:
        char_emb_var.set_weights([emb_mat])
    # Character embedding layer
    c_emb = char_emb_var(char_input)

    # Character embedding network
    conv_results_list = []
    for filter_width in filter_widths:
        conv_results_list.append(Conv2D(char_embedding_dim,
                                        (1, filter_width),
                                        padding='same')(c_emb))
    # units = Concatenate(conv_results_list, axis=3)
    units = Lambda(lambda x: K.concatenate(x, axis=3))(conv_results_list)
    units = Lambda(lambda x: K.max(x, axis=2))(units)
    if highway_on_top:
        sigmoid_gate = Dense(char_embedding_dim * len(filter_widths))(units)
        sigmoid_gate = Activation('sigmoid')(sigmoid_gate)
        char_embedding_dim * len(filter_widths)
        deeper_units = Dense(char_embedding_dim * len(filter_widths))(units)
        units = Multiply()([sigmoid_gate, deeper_units]) + \
                Multiply()([Lambda(lambda x: K.constant(1., shape=K.int_shape(x))-x)(sigmoid_gate), units])
        units = Activation('relu')(units)
    return units

class CharEmbeddingCNN(Layer):

    def __init__(self,
                 n_characters: int = None,
                 emb_mat: np.array = None,
                 char_embedding_dim: int = None,
                 filter_widths=(3, 4, 5, 7),
                 highway_on_top=False,
                 **kwargs):
        self.n_characters=n_characters,
        self.char_embedding_dim = char_embedding_dim,
        self.filter_widths = filter_widths,
        self.highway_on_top = highway_on_top
        super(CharEmbeddingCNN, self).__init__(**kwargs)

        self.char_emb_var = Embedding(n_characters,
                                 char_embedding_dim)
        if emb_mat is not None:
            self.char_emb_var.set_weights([emb_mat])

        self.conv2d_layers = []
        for filter_width in filter_widths:
            self.conv2d_layers.append(Conv2D(char_embedding_dim,
                                  (1, filter_width),
                                  padding='same'))

    def build(self, input_shape):
        pass

    def call(self, x):
        c_emb = self.char_emb_var(x)

        conv_results_list = []
        for cl in self.conv2d_layers:
            conv_results_list.append(cl(c_emb))

        units = Lambda(lambda x: K.concatenate(x, axis=3))(conv_results_list)
        units = Lambda(lambda x: K.max(x, axis=2))(units)
        return units

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.char_embedding_dim * len(self.filter_widths))


def char_embedding_cnn(x,
                     n_characters: int = None,
                     emb_mat: np.array = None,
                     char_embedding_dim: int = None,
                     filter_widths=(3, 4, 5, 7),
                     highway_on_top=False):
    char_emb_var = Embedding(n_characters,
                                  char_embedding_dim)
    if emb_mat is not None:
        char_emb_var.set_weights([emb_mat])

    conv2d_layers = []
    for filter_width in filter_widths:
        conv2d_layers.append(Conv2D(char_embedding_dim,
                                         (1, filter_width),
                                         padding='same'))
    c_emb = char_emb_var(x)

    conv_results_list = []
    for cl in conv2d_layers:
        conv_results_list.append(cl(c_emb))

    units = Lambda(lambda x: K.concatenate(x, axis=3))(conv_results_list)
    units = Lambda(lambda x: K.max(x, axis=2))(units)
    model = Model(input=x, outputs=units)
    return model
