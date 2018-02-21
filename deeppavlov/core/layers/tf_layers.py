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
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from typing import List


def stacked_cnn(units: tf.Tensor,
                n_hidden_list: List,
                filter_width=3,
                use_batch_norm=False,
                use_dilation=False,
                training_ph=None):
    """ Number of convolutional layers stacked on top of each other

    Args:
        units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
        n_hidden_list: list with number of hidden units at the ouput of each layer
        filter_width: width of the kernel in tokens
        use_batch_norm: whether to use batch normalization between layers
        use_dilation: use power of 2 dilation scheme [1, 2, 4, 8 .. ] for layers 1, 2, 3, 4 ...
        training_ph: boolean placeholder determining whether is training phase now or not.
            It is used only for batch normalization to determine whether to use
            current batch average (std) or memory stored average (std)
    Returns:
        units: tensor at the output of the last convolutional layer
    """

    for n_layer, n_hidden in enumerate(n_hidden_list):
        if use_dilation:
            dilation_rate = 2**n_layer
        else:
            dilation_rate = 1
        units = tf.layers.conv1d(units,
                                 n_hidden,
                                 filter_width,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer=xavier_initializer())
        if use_batch_norm:
            assert training_ph is not None
            units = tf.layers.batch_normalization(units, training=training_ph)
        units = tf.nn.relu(units)
    return units


def dense_convolutional_network(units: tf.Tensor,
                                n_hidden_list: List,
                                filter_width=3,
                                use_dilation=False,
                                use_batch_norm=False,
                                training_ph=None):
    """ Densely connected convolutional layers. Based on the paper:
        [Gao 17] https://arxiv.org/abs/1608.06993

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden_list: list with number of hidden units at the ouput of each layer
            filter_width: width of the kernel in tokens
            use_batch_norm: whether to use batch normalization between layers
            use_dilation: use power of 2 dilation scheme [1, 2, 4, 8 .. ] for layers 1, 2, 3, 4 ...
            training_ph: boolean placeholder determining whether is training phase now or not.
                It is used only for batch normalization to determine whether to use
                current batch average (std) or memory stored average (std)
        Returns:
            units: tensor at the output of the last convolutional layer
                with dimensionality [None, n_tokens, n_hidden_list[-1]]
        """
    units_list = [units]
    for n_layer, n_filters in enumerate(n_hidden_list):
        total_units = tf.concat(units_list, axis=-1)
        if use_dilation:
            dilation_rate = 2**n_layer
        else:
            dilation_rate = 1
        units = tf.layers.conv1d(total_units,
                                 n_filters,
                                 filter_width,
                                 dilation_rate=dilation_rate,
                                 padding='same',
                                 kernel_initializer=xavier_initializer())
        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training=training_ph)
        units = tf.nn.relu(units)
        units_list.append(units)
    return units


def stacked_rnn(units: tf.Tensor,
                n_hidden_list: List,
                cell_type='gru',
                use_peepholes=False,
                name='RNN_layer'):
    """ Stackted recurrent neural networks GRU or LSTM

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden_list: list with number of hidden units at the ouput of each layer
            cell_type: 'lstm' or 'gru'
            use_peepholes: whether to use peephole connections (only 'lstm' case affected)
            name: what variable_scope to use for the network parameters
        Returns:
            units: tensor at the output of the last recurrent layer
                   with dimensionality [None, n_tokens, n_hidden_list[-1]]
    """
    for n, n_hidden in enumerate(n_hidden_list):
        with tf.variable_scope(name + '_' + str(n)):
            if cell_type == 'gru':
                forward_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
                backward_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
            elif cell_type == 'lstm':
                forward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes)
                backward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes)
            else:
                raise RuntimeError('cell_type must be either gru or lstm')

            (rnn_output_fw, rnn_output_bw), _ = \
                tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                backward_cell,
                                                units,
                                                dtype=tf.float32)
            units = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)
    return units


def u_shape(units: tf.Tensor,
            n_hidden_list: List,
            filter_width=7,
            use_batch_norm=False,
            training_ph=None):
    """ Network architecture inspired by One Hundred layer Tiramisu.
        https://arxiv.org/abs/1611.09326. U-Net like.

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden_list: list with number of hidden units at the ouput of each layer
            filter_width: width of the kernel in tokens
            use_batch_norm: whether to use batch normalization between layers
            training_ph: boolean placeholder determining whether is training phase now or not.
                It is used only for batch normalization to determine whether to use
                current batch average (std) or memory stored average (std)
        Returns:
            units: tensor at the output of the last convolutional layer
                    with dimensionality [None, n_tokens, n_hidden_list[-1]]
    """

    # Bread Crumbs
    units_for_skip_conn = []
    conv_net_params = {'filter_width': filter_width,
                       'use_batch_norm': use_batch_norm,
                       'training_ph': training_ph}

    # Go down the rabbit hole
    for n_hidden in range(n_hidden_list):

        units = stacked_cnn(units, [n_hidden], **conv_net_params)
        units_for_skip_conn.append(units)
        units = tf.layers.max_pooling1d(units, pool_size=2, strides=2, padding='same')

    units = stacked_cnn(units, [n_hidden], **conv_net_params)

    # Up to the sun light
    for down_step, n_hidden in enumerate(n_hidden_list):
        units = tf.expand_dims(units, axis=2)
        units = tf.layers.conv2d_transpose(units, n_hidden, filter_width, strides=(2, 1), padding='same')
        units = tf.squeeze(units, axis=2)

        # Skip connection
        units = units_for_skip_conn[-(down_step + 1)] + units

        units = stacked_cnn(units, **conv_net_params)
    return units


def stacked_highway_cnn(units: tf.Tensor,
                        n_hidden_list,
                        filter_width=3,
                        use_batch_norm=False,
                        use_dilation=False,
                        training_ph=None):
    """Highway convolutional network. Skip connection with gating
        mechanism.

    Args:
        units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
        n_hidden_list: list with number of hidden units at the output of each layer
        filter_width: width of the kernel in tokens
        use_batch_norm: whether to use batch normalization between layers
        use_dilation: use power of 2 dilation scheme [1, 2, 4, 8 .. ] for layers 1, 2, 3, 4 ...
        training_ph: boolean placeholder determining whether is training phase now or not.
            It is used only for batch normalization to determine whether to use
            current batch average (std) or memory stored average (std)
    Returns:
        units: tensor at the output of the last convolutional layer
                with dimensionality [None, n_tokens, n_hidden_list[-1]]
    """
    for n_layer, n_hidden in enumerate(n_hidden_list):
        if use_dilation:
            dilation_rate = 2**n_layer
        else:
            dilation_rate = 1
        units = tf.layers.conv1d(units,
                                 n_hidden,
                                 filter_width,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer=xavier_initializer())
        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training=training_ph)
        sigmoid_gate = tf.layers.dense(input_units, 1, activation=tf.sigmoid, kernel_initializer=xavier_initializer())
        input_units = sigmoid_gate * input_units + (1 - sigmoid_gate) * units
        input_units = tf.nn.relu(input_units)
    units = input_units
    return units


def embedding_layer(token_indices=None,
                    token_embedding_matrix=None,
                    n_tokens=None,
                    token_embedding_dim=None,
                    name: str=None,
                    trainable=True):
    """ Token embedding layer. Create matrix of for token embeddings.
        Can be initialized with given matrix (for example pre-trained
        with word2ve algorithm

    Args:
        token_indices: token indices tensor of type tf.int32
        token_embedding_matrix: matrix of embeddings with dimensionality
            [n_tokens, embeddings_dimension]
        n_tokens: total number of unique tokens
        token_embedding_dim: dimensionality of embeddings, typical 100..300
        name: embedding matrix name (variable name)
        trainable: whether to set the matrix trainable or not

    Returns:
        embedded_tokens: tf tensor of size [B, T, E], where B - batch size
            T - number of tokens, E - token_embedding_dim
    """
    if token_embedding_matrix is not None:
        tok_mat = token_embedding_matrix
        if trainable:
            Warning('Matrix of embeddings is passed to the embedding_layer, '
                    'possibly there is a pre-trained embedding matrix. '
                    'Embeddings paramenters are set to Trainable!')
    else:
        tok_mat = np.random.randn(n_tokens, token_embedding_dim).astype(np.float32) / np.sqrt(token_embedding_dim)
    tok_emb_mat = tf.Variable(tok_mat, name=name, trainable=trainable)
    embedded_tokens = tf.nn.embedding_lookup(tok_emb_mat, token_indices)
    return embedded_tokens


def character_embedding_network(char_placeholder: tf.Tensor,
                                n_characters: int,
                                char_embedding_dim: int,
                                filter_width=7):
    """ Characters to vector. Every sequence of characters (token)
        is embedded to vector space with dimensionality char_embedding_dim
        Convolution plus max_pooling is used to obtain vector representations
        of words.

    Args:
        char_placeholder: placeholder of int32 type with dimensionality [B, T, C]
            B - batch size (can be None)
            T - Number of tokens (can be None)
            C - number of characters (can be None)
        n_characters: total number of unique characters
        char_embedding_dim: dimensionality of characters embeddings
        filter_width: width of kernel in convolutional embedding network

    Returns:
        embeddings: tf.Tensor with dimensionality [B, T, F],
            where F is dimensionality of embeddings
    """
    char_emb_mat = np.random.randn(n_characters, char_embedding_dim).astype(np.float32) / np.sqrt(char_embedding_dim)
    char_emb_var = tf.Variable(char_emb_mat, trainable=True)
    with tf.variable_scope('Char_Emb_Network'):
        # Character embedding layer
        c_emb = tf.nn.embedding_lookup(char_emb_var, char_placeholder)

        # Character embedding network
        char_conv = tf.layers.conv2d(c_emb, char_embedding_dim, (1, filter_width), padding='same', name='char_conv')
        embeddings = tf.reduce_max(char_conv, axis=2)
    return embeddings


if __name__ == '__main__':
    batch_size = 4
    tokens = 16
    features = 50
    var = tf.Variable(np.random.randn(batch_size, tokens, features).astype(np.float32))
    u_shape(var, 2, 100, 2, 3)
    stacked_highway_cnn(var, [100, 200])
    stacked_cnn(var, [100, 200])
    stacked_rnn(var, [100, 200], 'gru')
    stacked_rnn(var, [100, 200], 'lstm', use_peepholes=True)
