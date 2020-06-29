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

from logging import getLogger
from typing import List, Union

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.check_gpu import check_gpu_existence

log = getLogger(__name__)

INITIALIZER = tf.orthogonal_initializer


# INITIALIZER = xavier_initializer


def stacked_cnn(units: tf.Tensor,
                n_hidden_list: List,
                filter_width=3,
                use_batch_norm=False,
                use_dilation=False,
                training_ph=None,
                add_l2_losses=False):
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
        add_l2_losses: whether to add l2 losses on network kernels to
                tf.GraphKeys.REGULARIZATION_LOSSES or not

    Returns:
        units: tensor at the output of the last convolutional layer
    """
    l2_reg = tf.nn.l2_loss if add_l2_losses else None
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
                                 kernel_initializer=INITIALIZER(),
                                 kernel_regularizer=l2_reg)
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
                                 kernel_initializer=INITIALIZER())
        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training=training_ph)
        units = tf.nn.relu(units)
        units_list.append(units)
    return units


def bi_rnn(units: tf.Tensor,
           n_hidden: Union[List, int],
           cell_type='gru',
           seq_lengths=None,
           trainable_initial_states=False,
           use_peepholes=False,
           name='Bi-'):
    """ Bi directional recurrent neural network. GRU or LSTM

        Args:
            units: a tensorflow tensor with dimensionality [None, n_tokens, n_features]
            n_hidden: list with number of hidden units in the output of each layer if
                cell_type is 'lstm' and int if cell_type is 'gru'.
            seq_lengths: length of sequences for different length sequences in batch
                can be None for maximum length as a length for every sample in the batch
            cell_type: 'lstm' or 'gru'
            trainable_initial_states: whether to create a special trainable variable
                to initialize the hidden states of the network or use just zeros
            use_peepholes: whether to use peephole connections (only 'lstm' case affected)
            name: what variable_scope to use for the network parameters

        Returns:
            units: a tuple of tensors at the output of the last recurrent layer
                with dimensionality [None, n_tokens, n_hidden[-1]] if cell_type is 'lstm' and
                with dimensionality [None, n_tokens, n_hidden] if cell_type is 'gru'.
                The tensors contain the outputs of forward and backward passes of
                the birnn correspondingly.
            last_units: tensor of last hidden states for GRU and tuple
                of last hidden stated and last cell states for LSTM
                dimensionality of cell states and hidden states are
                similar and equal to [B x 2 * H], where B - batch
                size and H is number of hidden units
    """

    with tf.variable_scope(name + '_' + cell_type.upper()):
        if cell_type == 'gru':
            forward_cell = tf.nn.rnn_cell.GRUCell(n_hidden, kernel_initializer=INITIALIZER())
            backward_cell = tf.nn.rnn_cell.GRUCell(n_hidden, kernel_initializer=INITIALIZER())
            if trainable_initial_states:
                initial_state_fw = tf.tile(tf.get_variable('init_fw_h', [1, n_hidden]), (tf.shape(units)[0], 1))
                initial_state_bw = tf.tile(tf.get_variable('init_bw_h', [1, n_hidden]), (tf.shape(units)[0], 1))
            else:
                initial_state_fw = initial_state_bw = None
        elif cell_type == 'lstm':
            forward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes, initializer=INITIALIZER())
            backward_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=use_peepholes, initializer=INITIALIZER())
            if trainable_initial_states:
                initial_state_fw = tf.nn.rnn_cell.LSTMStateTuple(
                    tf.tile(tf.get_variable('init_fw_c', [1, n_hidden]), (tf.shape(units)[0], 1)),
                    tf.tile(tf.get_variable('init_fw_h', [1, n_hidden]), (tf.shape(units)[0], 1)))
                initial_state_bw = tf.nn.rnn_cell.LSTMStateTuple(
                    tf.tile(tf.get_variable('init_bw_c', [1, n_hidden]), (tf.shape(units)[0], 1)),
                    tf.tile(tf.get_variable('init_bw_h', [1, n_hidden]), (tf.shape(units)[0], 1)))
            else:
                initial_state_fw = initial_state_bw = None
        else:
            raise RuntimeError('cell_type must be either "gru" or "lstm"s')
        (rnn_output_fw, rnn_output_bw), (fw, bw) = \
            tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                            backward_cell,
                                            units,
                                            dtype=tf.float32,
                                            sequence_length=seq_lengths,
                                            initial_state_fw=initial_state_fw,
                                            initial_state_bw=initial_state_bw)
    kernels = [var for var in forward_cell.trainable_variables +
               backward_cell.trainable_variables if 'kernel' in var.name]
    for kernel in kernels:
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(kernel))
    return (rnn_output_fw, rnn_output_bw), (fw, bw)


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
                                 kernel_initializer=INITIALIZER())
        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training=training_ph)
        sigmoid_gate = tf.layers.dense(input_units, 1, activation=tf.sigmoid, kernel_initializer=INITIALIZER())
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
        char_placeholder: placeholder of int32 type with dimensionality [B, T, C]
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
        embeddings: tf.Tensor with dimensionality [B, T, F],
            where F is dimensionality of embeddings
    """
    if emb_mat is None:
        emb_mat = np.random.randn(n_characters, char_embedding_dim).astype(np.float32) / np.sqrt(char_embedding_dim)
    else:
        char_embedding_dim = emb_mat.shape[1]
    char_emb_var = tf.Variable(emb_mat, trainable=True)
    with tf.variable_scope('Char_Emb_Network'):
        # Character embedding layer
        c_emb = tf.nn.embedding_lookup(char_emb_var, char_placeholder)

        # Character embedding network
        conv_results_list = []
        for filter_width in filter_widths:
            conv_results_list.append(tf.layers.conv2d(c_emb,
                                                      char_embedding_dim,
                                                      (1, filter_width),
                                                      padding='same',
                                                      kernel_initializer=INITIALIZER))
        units = tf.concat(conv_results_list, axis=3)
        units = tf.reduce_max(units, axis=2)
        if highway_on_top:
            sigmoid_gate = tf.layers.dense(units,
                                           1,
                                           activation=tf.sigmoid,
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=tf.nn.l2_loss)
            deeper_units = tf.layers.dense(units,
                                           tf.shape(units)[-1],
                                           kernel_initializer=INITIALIZER,
                                           kernel_regularizer=tf.nn.l2_loss)
            units = sigmoid_gate * units + (1 - sigmoid_gate) * deeper_units
            units = tf.nn.relu(units)
    return units


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
    query = tf.layers.dense(units_pairs, n_hidden, activation=tf.tanh, kernel_initializer=INITIALIZER())
    attention = tf.nn.softmax(tf.layers.dense(query, 1), dim=2)
    attended_units = tf.reduce_sum(attention * expand_tile(units, 1), axis=2)
    output = tf.layers.dense(attended_units, n_output_features, activation, kernel_initializer=INITIALIZER())
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
    queries = tf.layers.dense(expand_tile(units, 1), n_hidden, kernel_initializer=INITIALIZER())
    keys = tf.layers.dense(expand_tile(units, 2), n_hidden, kernel_initializer=INITIALIZER())
    scores = tf.reduce_sum(queries * keys, axis=3, keep_dims=True)
    attention = tf.nn.softmax(scores, dim=2)
    attended_units = tf.reduce_sum(attention * expand_tile(units, 1), axis=2)
    output = tf.layers.dense(attended_units, n_output_features, activation, kernel_initializer=INITIALIZER())
    return output


def cudnn_gru(units, n_hidden, n_layers=1, trainable_initial_states=False,
              seq_lengths=None, input_initial_h=None, name='cudnn_gru', reuse=False):
    """ Fast CuDNN GRU implementation

    Args:
        units: tf.Tensor with dimensions [B x T x F], where
            B - batch size
            T - number of tokens
            F - features

        n_hidden: dimensionality of hidden state
        trainable_initial_states: whether to create a special trainable variable
            to initialize the hidden states of the network or use just zeros
        seq_lengths: tensor of sequence lengths with dimension [B]
        n_layers: number of layers
        input_initial_h: initial hidden state, tensor
        name: name of the variable scope to use
        reuse:whether to reuse already initialized variable

    Returns:
        h - all hidden states along T dimension,
            tf.Tensor with dimensionality [B x T x F]
        h_last - last hidden state, tf.Tensor with dimensionality [B x H]
    """
    with tf.variable_scope(name, reuse=reuse):
        gru = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=n_layers,
                                            num_units=n_hidden)

        if trainable_initial_states:
            init_h = tf.get_variable('init_h', [n_layers, 1, n_hidden])
            init_h = tf.tile(init_h, (1, tf.shape(units)[0], 1))
        else:
            init_h = tf.zeros([n_layers, tf.shape(units)[0], n_hidden])

        initial_h = input_initial_h or init_h

        h, h_last = gru(tf.transpose(units, (1, 0, 2)), (initial_h,))
        h = tf.transpose(h, (1, 0, 2))
        h_last = tf.squeeze(h_last, axis=0)[-1]  # extract last layer state

        # Extract last states if they are provided
        if seq_lengths is not None:
            indices = tf.stack([tf.range(tf.shape(h)[0]), seq_lengths - 1], axis=1)
            h_last = tf.gather_nd(h, indices)

        return h, h_last


def cudnn_compatible_gru(units, n_hidden, n_layers=1, trainable_initial_states=False,
                         seq_lengths=None, input_initial_h=None, name='cudnn_gru', reuse=False):
    """ CuDNN Compatible GRU implementation.
        It should be used to load models saved with CudnnGRUCell to run on CPU.

        Args:
            units: tf.Tensor with dimensions [B x T x F], where
                B - batch size
                T - number of tokens
                F - features

            n_hidden: dimensionality of hidden state
            trainable_initial_states: whether to create a special trainable variable
                to initialize the hidden states of the network or use just zeros
            seq_lengths: tensor of sequence lengths with dimension [B]
            n_layers: number of layers
            input_initial_h: initial hidden state, tensor
            name: name of the variable scope to use
            reuse:whether to reuse already initialized variable

        Returns:
            h - all hidden states along T dimension,
                tf.Tensor with dimensionality [B x T x F]
            h_last - last hidden state, tf.Tensor with dimensionality [B x H]
        """
    with tf.variable_scope(name, reuse=reuse):

        if trainable_initial_states:
            init_h = tf.get_variable('init_h', [n_layers, 1, n_hidden])
            init_h = tf.tile(init_h, (1, tf.shape(units)[0], 1))
        else:
            init_h = tf.zeros([n_layers, tf.shape(units)[0], n_hidden])

        initial_h = input_initial_h or init_h

        with tf.variable_scope('cudnn_gru', reuse=reuse):
            def single_cell(): return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(n_hidden)

            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(n_layers)])

            units = tf.transpose(units, (1, 0, 2))

            h, h_last = tf.nn.dynamic_rnn(cell=cell, inputs=units, time_major=True,
                                          initial_state=tuple(tf.unstack(initial_h, axis=0)))
            h = tf.transpose(h, (1, 0, 2))

            h_last = h_last[-1]  # h_last is tuple: n_layers x batch_size x n_hidden

            # Extract last states if they are provided
            if seq_lengths is not None:
                indices = tf.stack([tf.range(tf.shape(h)[0]), seq_lengths - 1], axis=1)
                h_last = tf.gather_nd(h, indices)

            return h, h_last


def cudnn_gru_wrapper(units, n_hidden, n_layers=1, trainable_initial_states=False,
                      seq_lengths=None, input_initial_h=None, name='cudnn_gru', reuse=False):
    if check_gpu_existence():
        return cudnn_gru(units, n_hidden, n_layers, trainable_initial_states,
                         seq_lengths, input_initial_h, name, reuse)

    log.info('\nWarning! tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell is used. '
             'It is okay for inference mode, but '
             'if you train your model with this cell it could NOT be used with '
             'tf.contrib.cudnn_rnn.CudnnGRUCell later. '
             )

    return cudnn_compatible_gru(units, n_hidden, n_layers, trainable_initial_states,
                                seq_lengths, input_initial_h, name, reuse)


def cudnn_lstm(units, n_hidden, n_layers=1, trainable_initial_states=None, seq_lengths=None, initial_h=None,
               initial_c=None, name='cudnn_lstm', reuse=False):
    """ Fast CuDNN LSTM implementation

        Args:
            units: tf.Tensor with dimensions [B x T x F], where
                B - batch size
                T - number of tokens
                F - features
            n_hidden: dimensionality of hidden state
            n_layers: number of layers
            trainable_initial_states: whether to create a special trainable variable
                to initialize the hidden states of the network or use just zeros
            seq_lengths: tensor of sequence lengths with dimension [B]
            initial_h: optional initial hidden state, masks trainable_initial_states
                if provided
            initial_c: optional initial cell state, masks trainable_initial_states
                if provided
            name: name of the variable scope to use
            reuse:whether to reuse already initialized variable


        Returns:
            h - all hidden states along T dimension,
                tf.Tensor with dimensionality [B x T x F]
            h_last - last hidden state, tf.Tensor with dimensionality [B x H]
                where H - number of hidden units
            c_last - last cell state, tf.Tensor with dimensionality [B x H]
                where H - number of hidden units
        """
    with tf.variable_scope(name, reuse=reuse):
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=n_layers,
                                              num_units=n_hidden)
        if trainable_initial_states:
            init_h = tf.get_variable('init_h', [n_layers, 1, n_hidden])
            init_h = tf.tile(init_h, (1, tf.shape(units)[0], 1))
            init_c = tf.get_variable('init_c', [n_layers, 1, n_hidden])
            init_c = tf.tile(init_c, (1, tf.shape(units)[0], 1))
        else:
            init_h = init_c = tf.zeros([n_layers, tf.shape(units)[0], n_hidden])

        initial_h = initial_h or init_h
        initial_c = initial_c or init_c

        h, (h_last, c_last) = lstm(tf.transpose(units, (1, 0, 2)), (initial_h, initial_c))
        h = tf.transpose(h, (1, 0, 2))
        h_last = h_last[-1]
        c_last = c_last[-1]

        # Extract last states if they are provided
        if seq_lengths is not None:
            indices = tf.stack([tf.range(tf.shape(h)[0]), seq_lengths - 1], axis=1)
            h_last = tf.gather_nd(h, indices)

        return h, (h_last, c_last)


def cudnn_compatible_lstm(units, n_hidden, n_layers=1, trainable_initial_states=None, seq_lengths=None, initial_h=None,
                          initial_c=None, name='cudnn_lstm', reuse=False):
    """ CuDNN Compatible LSTM implementation.
        It should be used to load models saved with CudnnLSTMCell to run on CPU.

        Args:
            units: tf.Tensor with dimensions [B x T x F], where
                B - batch size
                T - number of tokens
                F - features
            n_hidden: dimensionality of hidden state
            n_layers: number of layers
            trainable_initial_states: whether to create a special trainable variable
                to initialize the hidden states of the network or use just zeros
            seq_lengths: tensor of sequence lengths with dimension [B]
            initial_h: optional initial hidden state, masks trainable_initial_states
                if provided
            initial_c: optional initial cell state, masks trainable_initial_states
                if provided
            name: name of the variable scope to use
            reuse:whether to reuse already initialized variable


        Returns:
            h - all hidden states along T dimension,
                tf.Tensor with dimensionality [B x T x F]
            h_last - last hidden state, tf.Tensor with dimensionality [B x H]
                where H - number of hidden units
            c_last - last cell state, tf.Tensor with dimensionality [B x H]
                where H - number of hidden units
        """

    with tf.variable_scope(name, reuse=reuse):
        if trainable_initial_states:
            init_h = tf.get_variable('init_h', [n_layers, 1, n_hidden])
            init_h = tf.tile(init_h, (1, tf.shape(units)[0], 1))
            init_c = tf.get_variable('init_c', [n_layers, 1, n_hidden])
            init_c = tf.tile(init_c, (1, tf.shape(units)[0], 1))
        else:
            init_h = init_c = tf.zeros([n_layers, tf.shape(units)[0], n_hidden])

        initial_h = initial_h or init_h
        initial_c = initial_c or init_c

        with tf.variable_scope('cudnn_lstm', reuse=reuse):
            def single_cell(): return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(n_hidden)

            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(n_layers)])

            units = tf.transpose(units, (1, 0, 2))

            init = tuple([tf.nn.rnn_cell.LSTMStateTuple(ic, ih) for ih, ic in
                          zip(tf.unstack(initial_h, axis=0), tf.unstack(initial_c, axis=0))])

            h, state = tf.nn.dynamic_rnn(cell=cell, inputs=units, time_major=True, initial_state=init)

            h = tf.transpose(h, (1, 0, 2))
            h_last = state[-1].h
            c_last = state[-1].c

            # Extract last states if they are provided
            if seq_lengths is not None:
                indices = tf.stack([tf.range(tf.shape(h)[0]), seq_lengths - 1], axis=1)
                h_last = tf.gather_nd(h, indices)

            return h, (h_last, c_last)


def cudnn_lstm_wrapper(units, n_hidden, n_layers=1, trainable_initial_states=None, seq_lengths=None, initial_h=None,
                       initial_c=None, name='cudnn_lstm', reuse=False):
    if check_gpu_existence():
        return cudnn_lstm(units, n_hidden, n_layers, trainable_initial_states,
                          seq_lengths, initial_h, initial_c, name, reuse)

    log.info('\nWarning! tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell is used. '
             'It is okay for inference mode, but '
             'if you train your model with this cell it could NOT be used with '
             'tf.contrib.cudnn_rnn.CudnnLSTMCell later. '
             )

    return cudnn_compatible_lstm(units, n_hidden, n_layers, trainable_initial_states,
                                 seq_lengths, initial_h, initial_c, name, reuse)


def cudnn_bi_gru(units,
                 n_hidden,
                 seq_lengths=None,
                 n_layers=1,
                 trainable_initial_states=False,
                 name='cudnn_bi_gru',
                 reuse=False):
    """ Fast CuDNN Bi-GRU implementation

    Args:
        units: tf.Tensor with dimensions [B x T x F], where
            B - batch size
            T - number of tokens
            F - features
        n_hidden: dimensionality of hidden state
        seq_lengths: number of tokens in each sample in the batch
        n_layers: number of layers
        trainable_initial_states: whether to create a special trainable variable
                to initialize the hidden states of the network or use just zeros
        name: name of the variable scope to use
        reuse:whether to reuse already initialized variable


    Returns:
        h - all hidden states along T dimension,
            tf.Tensor with dimensionality [B x T x F]
        h_last - last hidden state, tf.Tensor with dimensionality [B x H * 2]
            where H - number of hidden units
    """

    with tf.variable_scope(name, reuse=reuse):
        if seq_lengths is None:
            seq_lengths = tf.ones([tf.shape(units)[0]], dtype=tf.int32) * tf.shape(units)[1]
        with tf.variable_scope('Forward'):
            h_fw, h_last_fw = cudnn_gru_wrapper(units,
                                                n_hidden,
                                                n_layers=n_layers,
                                                trainable_initial_states=trainable_initial_states,
                                                seq_lengths=seq_lengths,
                                                reuse=reuse)

        with tf.variable_scope('Backward'):
            reversed_units = tf.reverse_sequence(units, seq_lengths=seq_lengths, seq_dim=1, batch_dim=0)
            h_bw, h_last_bw = cudnn_gru_wrapper(reversed_units,
                                                n_hidden,
                                                n_layers=n_layers,
                                                trainable_initial_states=trainable_initial_states,
                                                seq_lengths=seq_lengths,
                                                reuse=reuse)
            h_bw = tf.reverse_sequence(h_bw, seq_lengths=seq_lengths, seq_dim=1, batch_dim=0)

    return (h_fw, h_bw), (h_last_fw, h_last_bw)


def cudnn_bi_lstm(units,
                  n_hidden,
                  seq_lengths=None,
                  n_layers=1,
                  trainable_initial_states=False,
                  name='cudnn_bi_gru',
                  reuse=False):
    """ Fast CuDNN Bi-LSTM implementation

    Args:
        units: tf.Tensor with dimensions [B x T x F], where
            B - batch size
            T - number of tokens
            F - features
        n_hidden: dimensionality of hidden state
        seq_lengths: number of tokens in each sample in the batch
        n_layers: number of layers
        trainable_initial_states: whether to create a special trainable variable
            to initialize the hidden states of the network or use just zeros
        name: name of the variable scope to use
        reuse:whether to reuse already initialized variable

    Returns:
        h - all hidden states along T dimension,
            tf.Tensor with dimensionality [B x T x F]
        h_last - last hidden state, tf.Tensor with dimensionality [B x H * 2]
            where H - number of hidden units
        c_last - last cell state, tf.Tensor with dimensionality [B x H * 2]
            where H - number of hidden units
        """
    with tf.variable_scope(name, reuse=reuse):
        if seq_lengths is None:
            seq_lengths = tf.ones([tf.shape(units)[0]], dtype=tf.int32) * tf.shape(units)[1]
        with tf.variable_scope('Forward'):
            h_fw, (h_fw_last, c_fw_last) = cudnn_lstm_wrapper(units,
                                                              n_hidden,
                                                              n_layers=n_layers,
                                                              trainable_initial_states=trainable_initial_states,
                                                              seq_lengths=seq_lengths)

        with tf.variable_scope('Backward'):
            reversed_units = tf.reverse_sequence(units, seq_lengths=seq_lengths, seq_dim=1, batch_dim=0)
            h_bw, (h_bw_last, c_bw_last) = cudnn_lstm_wrapper(reversed_units,
                                                              n_hidden,
                                                              n_layers=n_layers,
                                                              trainable_initial_states=trainable_initial_states,
                                                              seq_lengths=seq_lengths)

            h_bw = tf.reverse_sequence(h_bw, seq_lengths=seq_lengths, seq_dim=1, batch_dim=0)
        return (h_fw, h_bw), ((h_fw_last, c_fw_last), (h_bw_last, c_bw_last))


def cudnn_stacked_bi_gru(units,
                         n_hidden,
                         seq_lengths=None,
                         n_stacks=2,
                         keep_prob=1.0,
                         concat_stacked_outputs=False,
                         trainable_initial_states=False,
                         name='cudnn_stacked_bi_gru',
                         reuse=False):
    """ Fast CuDNN Stacked Bi-GRU implementation

    Args:
        units: tf.Tensor with dimensions [B x T x F], where
            B - batch size
            T - number of tokens
            F - features
        n_hidden: dimensionality of hidden state
        seq_lengths: number of tokens in each sample in the batch
        n_stacks: number of stacked Bi-GRU
        keep_prob: dropout keep_prob between Bi-GRUs (intra-layer dropout)
        concat_stacked_outputs: return last Bi-GRU output or concat outputs from every Bi-GRU,
        trainable_initial_states: whether to create a special trainable variable
                to initialize the hidden states of the network or use just zeros
        name: name of the variable scope to use
        reuse: whether to reuse already initialized variable


    Returns:
        h - all hidden states along T dimension,
            tf.Tensor with dimensionality [B x T x ((n_hidden * 2) * n_stacks)]
    """
    if seq_lengths is None:
        seq_lengths = tf.ones([tf.shape(units)[0]], dtype=tf.int32) * tf.shape(units)[1]

    outputs = [units]

    with tf.variable_scope(name, reuse=reuse):
        for n in range(n_stacks):

            if n == 0:
                inputs = outputs[-1]
            else:
                inputs = variational_dropout(outputs[-1], keep_prob=keep_prob)

            (h_fw, h_bw), _ = cudnn_bi_gru(inputs, n_hidden, seq_lengths,
                                           n_layers=1,
                                           trainable_initial_states=trainable_initial_states,
                                           name='{}_cudnn_bi_gru'.format(n),
                                           reuse=reuse)

            outputs.append(tf.concat([h_fw, h_bw], axis=2))

    if concat_stacked_outputs:
        return tf.concat(outputs[1:], axis=2)

    return outputs[-1]


def variational_dropout(units, keep_prob, fixed_mask_dims=(1,)):
    """ Dropout with the same drop mask for all fixed_mask_dims

    Args:
        units: a tensor, usually with shapes [B x T x F], where
            B - batch size
            T - tokens dimension
            F - feature dimension
        keep_prob: keep probability
        fixed_mask_dims: in these dimensions the mask will be the same

    Returns:
        dropped units tensor
    """
    units_shape = tf.shape(units)
    noise_shape = [units_shape[n] for n in range(len(units.shape))]
    for dim in fixed_mask_dims:
        noise_shape[dim] = 1
    return tf.nn.dropout(units, rate=1 - keep_prob, noise_shape=noise_shape)
