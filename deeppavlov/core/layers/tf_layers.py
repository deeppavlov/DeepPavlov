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
            dilation_rate = 2 ** n_layer
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
            dilation_rate = 2 ** n_layer
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


def stacked_bi_rnn(units: tf.Tensor,
                   n_hidden_list: List,
                   cell_type='gru',
                   seq_lengths=None,
                   use_peepholes=False,
                   name='RNN_layer'):
    """ Stackted recurrent neural networks GRU or LSTM

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden_list: list with number of hidden units at the ouput of each layer
            seq_lengths: length of sequences for different length sequences in batch
                can be None for maximum length as a length for every sample in the batch
            cell_type: 'lstm' or 'gru'
            use_peepholes: whether to use peephole connections (only 'lstm' case affected)
            name: what variable_scope to use for the network parameters
        Returns:
            units: tensor at the output of the last recurrent layer
                with dimensionality [None, n_tokens, n_hidden_list[-1]]
            last_units: tensor of last hidden states for GRU and tuple
                of last hidden stated and last cell states for LSTM
                dimensionality of cell states and hidden states are
                similar and equal to [B x 2 * H], where B - batch
                size and H is number of hidden units
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

            (rnn_output_fw, rnn_output_bw), (fw, bw) = \
                tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                backward_cell,
                                                units,
                                                dtype=tf.float32,
                                                sequence_length=seq_lengths)
            units = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)
            if cell_type == 'gru':
                last_units = tf.concat([fw, bw], axis=1)
            else:
                (c_fw, h_fw), (c_bw, h_bw) = fw, bw
                c = tf.concat([c_fw, c_bw], axis=1)
                h = tf.concat([h_fw, h_bw], axis=1)
                last_units = (h, c)
    return units, last_units


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
    for n_hidden in n_hidden_list:
        units = stacked_cnn(units, [n_hidden], **conv_net_params)
        units_for_skip_conn.append(units)
        units = tf.layers.max_pooling1d(units, pool_size=2, strides=2, padding='same')

    units = stacked_cnn(units, [n_hidden], **conv_net_params)

    # Up to the sun light
    for down_step, n_hidden in enumerate(n_hidden_list[::-1]):
        units = tf.expand_dims(units, axis=2)
        units = tf.layers.conv2d_transpose(units, n_hidden, filter_width, strides=(2, 1), padding='same')
        units = tf.squeeze(units, axis=2)

        # Skip connection
        skip_units = units_for_skip_conn[-(down_step + 1)]
        if skip_units.get_shape().as_list()[-1] != n_hidden:
            skip_units = tf.layers.dense(skip_units, n_hidden)
        units = skip_units + units

        units = stacked_cnn(units, [n_hidden], **conv_net_params)
    return units


def stacked_highway_cnn(units: tf.Tensor,
                        n_hidden_list: List,
                        filter_width=3,
                        use_batch_norm=False,
                        use_dilation=False,
                        training_ph=None):
    """ Highway convolutional network. Skip connection with gating
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
        input_units = units
        # Projection if needed
        if input_units.get_shape().as_list()[-1] != n_hidden:
            input_units = tf.layers.dense(input_units, n_hidden)
        if use_dilation:
            dilation_rate = 2 ** n_layer
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
                    name: str = None,
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


def expand_tile(units, axis):
    """Expand and tile tensor along given axis
    Args:
        units: tf tensor with dimensions [batch_size, time_steps, n_input_features]
        axis: axis along which expand and tile. Must be 1 or 2

    """
    assert axis in (1, 2)
    n_time_steps = tf.shape(units)[1]
    repetitions = [1, 1, 1, 1]
    repetitions[axis] = n_time_steps
    return tf.tile(tf.expand_dims(units, axis), repetitions)


def additive_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """ Computes additive self attention for time series of vectors (with batch dimension)
        the formula: score(h_i, h_j) = <v, tanh(W_1 h_i + W_2 h_j)>
        v is a learnable vector of n_hidden dimensionality,
        W_1 and W_2 are learnable [n_hidden, n_input_features] matrices

    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output

    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
    """
    n_input_features = units.get_shape().as_list()[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    units_pairs = tf.concat([expand_tile(units, 1), expand_tile(units, 2)], 3)
    query = tf.layers.dense(units_pairs, n_hidden, activation=tf.tanh)
    attention = tf.nn.softmax(tf.layers.dense(query, 1), dim=2)
    attended_units = tf.reduce_sum(attention * expand_tile(units, 1), axis=2)
    output = tf.layers.dense(attended_units, n_output_features, activation)
    return output


def multiplicative_self_attention(units, n_hidden=None, n_output_features=None, activation=None):
    """ Computes multiplicative self attention for time series of vectors (with batch dimension)
        the formula: score(h_i, h_j) = <W_1 h_i,  W_2 h_j>,  W_1 and W_2 are learnable matrices
        with dimensionality [n_hidden, n_input_features], where <a, b> stands for a and b
        dot product

    Args:
        units: tf tensor with dimensionality [batch_size, time_steps, n_input_features]
        n_hidden: number of units in hidden representation of similarity measure
        n_output_features: number of features in output dense layer
        activation: activation at the output

    Returns:
        output: self attended tensor with dimensionality [batch_size, time_steps, n_output_features]
    """
    n_input_features = units.get_shape().as_list()[2]
    if n_hidden is None:
        n_hidden = n_input_features
    if n_output_features is None:
        n_output_features = n_input_features
    queries = tf.layers.dense(expand_tile(units, 1), n_hidden)
    keys = tf.layers.dense(expand_tile(units, 2), n_hidden)
    scores = tf.reduce_sum(queries * keys, axis=3, keep_dims=True)
    attention = tf.nn.softmax(scores, dim=2)
    attended_units = tf.reduce_sum(attention * expand_tile(units, 1), axis=2)
    output = tf.layers.dense(attended_units, n_output_features, activation)
    return output


def cudnn_gru(units, n_hidden, n_layers=1):
    """ Fast CuDNN GRU implementation

    Args:
        units: tf.Tensor with dimensions [B x T x F], where
            B - batch size
            T - number of tokens
            F - features
        n_hidden: dimensionality of hidden state
        n_layers: number of layers

    Returns:
        h - all hidden states along T dimension,
            tf.Tensor with dimensionality [B x T x F]
        h_last - last hidden state, tf.Tensor with dimensionality [B x H]
    """
    gru = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=n_layers,
                                        num_units=n_hidden,
                                        input_size=units.get_shape().as_list()[-1])
    param = tf.Variable(tf.random_uniform(
        [gru.params_size()], -0.1, 0.1), validate_shape=False)
    init_h = tf.zeros([1, tf.shape(units)[0], n_hidden])
    h, h_last = gru(tf.transpose(units, (1, 0, 2)), init_h, param)
    h = tf.transpose(h, (1, 0, 2))
    h_last = tf.squeeze(h_last, 0)
    return h, h_last


def cudnn_lstm(units, n_hidden, n_layers=1):
    """ Fast CuDNN LSTM implementation

        Args:
            units: tf.Tensor with dimensions [B x T x F], where
                B - batch size
                T - number of tokens
                F - features
            n_hidden: dimensionality of hidden state
            n_layers: number of layers

        Returns:
            h - all hidden states along T dimension,
                tf.Tensor with dimensionality [B x T x F]
            h_last - last hidden state, tf.Tensor with dimensionality [B x H]
                where H - number of hidden units
            c_last - last cell state, tf.Tensor with dimensionality [B x H]
                where H - number of hidden units
        """
    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=n_layers,
                                          num_units=n_hidden,
                                          input_size=units.get_shape().as_list()[-1])
    param = tf.Variable(tf.random_uniform(
        [lstm.params_size()], -0.1, 0.1), validate_shape=False)
    init_h = tf.zeros([1, tf.shape(units)[0], n_hidden])
    init_c = tf.zeros([1, tf.shape(units)[0], n_hidden])
    h, h_last, c_last = lstm(tf.transpose(units, (1, 0, 2)), init_h, init_c, param)
    h = tf.transpose(h, (1, 0, 2))
    h_last = tf.squeeze(h_last, 0)
    c_last = tf.squeeze(c_last, 0)
    return h, (h_last, c_last)


def cudnn_bi_gru(units, n_hidden, seq_lengths=None, n_layers=1):
    """ Fast CuDNN Bi-GRU implementation

    Args:
        units: tf.Tensor with dimensions [B x T x F], where
            B - batch size
            T - number of tokens
            F - features
        n_hidden: dimensionality of hidden state
        seq_lengths: number of tokens in each sample in the batch
        n_layers: number of layers

    Returns:
        h - all hidden states along T dimension,
            tf.Tensor with dimensionality [B x T x F]
        h_last - last hidden state, tf.Tensor with dimensionality [B x H * 2]
            where H - number of hidden units
    """
    gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=n_layers,
                                           num_units=n_hidden,
                                           input_size=units.get_shape().as_list()[-1])
    gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=n_layers,
                                           num_units=n_hidden,
                                           input_size=units.get_shape().as_list()[-1])

    param_fw = tf.Variable(tf.random_uniform(
        [gru_fw.params_size()], -0.1, 0.1), validate_shape=False)
    param_bw = tf.Variable(tf.random_uniform(
        [gru_bw.params_size()], -0.1, 0.1), validate_shape=False)

    init_h_fw = tf.zeros([1, tf.shape(units)[0], n_hidden])
    init_h_bw = tf.zeros([1, tf.shape(units)[0], n_hidden])

    h_fw, h_last_fw = gru_fw(tf.transpose(units, (1, 0, 2)), init_h_fw, param_fw)
    reversed_units = tf.reverse_sequence(units, seq_lengths=seq_lengths, seq_dim=1, batch_dim=0)
    h_bw, h_last_bw = gru_bw(tf.transpose(reversed_units, (1, 0, 2)), init_h_bw, param_bw)
    h_bw = tf.reverse_sequence(h_bw, seq_lengths=seq_lengths, seq_dim=0, batch_dim=1)

    h = tf.concat([h_fw, h_bw], axis=2)
    h_last = tf.concat([h_last_fw, h_last_bw], axis=2)
    h = tf.transpose(h, (1, 0, 2))
    h_last = tf.squeeze(h_last, 0)
    return h, h_last


def cudnn_bi_lstm(units, n_hidden, seq_lengths, n_layers=1):
    """ Fast CuDNN Bi-LSTM implementation

        Args:
            units: tf.Tensor with dimensions [B x T x F], where
                B - batch size
                T - number of tokens
                F - features
            n_hidden: dimensionality of hidden state
            n_layers: number of layers

        Returns:
            h - all hidden states along T dimension,
                tf.Tensor with dimensionality [B x T x F]
            h_last - last hidden state, tf.Tensor with dimensionality [B x H * 2]
                where H - number of hidden units
            c_last - last cell state, tf.Tensor with dimensionality [B x H * 2]
                where H - number of hidden units
        """
    lstm_fw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=n_layers,
                                             num_units=n_hidden,
                                             input_size=units.get_shape().as_list()[-1])
    lstm_bw = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=n_layers,
                                             num_units=n_hidden,
                                             input_size=units.get_shape().as_list()[-1])

    param_fw = tf.Variable(tf.random_uniform(
        [lstm_fw.params_size()], -0.1, 0.1), validate_shape=False)
    param_bw = tf.Variable(tf.random_uniform(
        [lstm_bw.params_size()], -0.1, 0.1), validate_shape=False)

    init_h_fw = tf.zeros([1, tf.shape(units)[0], n_hidden])
    init_h_bw = tf.zeros([1, tf.shape(units)[0], n_hidden])

    init_c_fw = tf.zeros([1, tf.shape(units)[0], n_hidden])
    init_c_bw = tf.zeros([1, tf.shape(units)[0], n_hidden])

    h_fw, h_last_fw, c_last_fw = lstm_fw(tf.transpose(units, (1, 0, 2)), init_h_fw, init_c_fw, param_fw)
    reversed_units = tf.reverse_sequence(units, seq_lengths=seq_lengths, seq_dim=1, batch_dim=0)
    h_bw, h_last_bw, c_last_bw = lstm_bw(tf.transpose(reversed_units, (1, 0, 2)), init_h_bw, init_c_bw, param_bw)
    h_bw = tf.reverse_sequence(h_bw, seq_lengths=seq_lengths, seq_dim=0, batch_dim=1)

    h = tf.concat([h_fw, h_bw], axis=2)
    h_last = tf.concat([h_last_fw, h_last_bw], axis=2)
    c_last = tf.concat([c_last_fw, c_last_bw], axis=2)

    h = tf.transpose(h, (1, 0, 2))
    h_last = tf.squeeze(h_last, 0)
    return h, (h_last, c_last)
