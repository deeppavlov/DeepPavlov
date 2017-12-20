import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import collections


def stacked_convolutions(input_units,
                         n_filters,
                         filter_width=3,
                         use_batch_norm=False,
                         use_dilation=False,
                         training_ph=None):
    units = input_units
    if n_filters is None:
        # If number of filters is not given the number of filters
        # will be equal to the number of input features
        n_filters = input_units.get_shape().as_list()[-1]
    # if isinstance(n_filters, collections.Iterable) and n_layers is not None:
    #     assert len(n_filters) == n_layers
    n_layers = len(n_filters)
    for n_layer in range(n_layers):
        if isinstance(n_filters, collections.Iterable):
            current_n_fileters = n_filters[n_layer]
        else:
            current_n_fileters = n_filters
        if use_dilation:
            dilation_rate = 2**n_layer
        else:
            dilation_rate = 1
        units = tf.layers.conv1d(units,
                                 current_n_fileters,
                                 filter_width,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer=xavier_initializer())
        if use_batch_norm:
            assert training_ph is not None
            units = tf.layers.batch_normalization(units, training=training_ph)
        units = tf.nn.relu(units)
    return units


def dense_convolutional_network(input_units,
                                n_filters=None,
                                n_layers=1,
                                filter_width=3,
                                use_dilation=False,
                                use_batch_norm=False,
                                training_ph=None):
    units = input_units
    if n_filters is None:
        # If number of filters is not given the number of filters
        # will be equal to the number of input features
        n_filters = input_units.get_shape().as_list()[-1]
    units_list = [units]
    for n_layer in range(n_layers):
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


def stacked_rnn(input_units,
                n_hidden_list,
                cell_type='gru'):
    units = input_units
    for n, n_h in enumerate(n_hidden_list):
        with tf.variable_scope('RNN_layer_' + str(n)):
            if cell_type == 'gru':
                forward_cell = tf.nn.rnn_cell.GRUCell(n_h)
                backward_cell = tf.nn.rnn_cell.GRUCell(n_h)
            elif cell_type == 'lstm':
                forward_cell = tf.nn.rnn_cell.LSTMCell(n_h)
                backward_cell = tf.nn.rnn_cell.LSTMCell(n_h)
            else:
                raise RuntimeError('cell_type must be either gru or lstm')

            (rnn_output_fw, rnn_output_bw), _ = \
                tf.nn.bidirectional_dynamic_rnn(forward_cell,
                                                backward_cell,
                                                units,
                                                dtype=tf.float32)

            # Dense layer on the top
            units = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)
    return units


def u_shape(input_units,
            rabbit_hole_depth=3,
            n_filters=None,
            n_layers=1,
            filter_width=3,
            use_batch_norm=False,
            use_dilation=False,
            training_ph=None):
    units = input_units
    conv_net_params = {'n_filters': n_filters,
                       'n_layers': n_layers,
                       'filter_width': filter_width,
                       'use_batch_norm': use_batch_norm,
                       'use_dilation': use_dilation,
                       'training_ph': training_ph}
    if n_filters is None:
        n_filters = input_units.get_shape().as_list()

    # Bread Crumbs
    units_for_skip_conn = list()

    # Go down the rabbit hole
    for down_step in range(rabbit_hole_depth):
        units = stacked_convolutions(units, **conv_net_params)
        units_for_skip_conn.append(units)
        units = tf.layers.max_pooling1d(units, pool_size=2, strides=2, padding='same')

    units = stacked_convolutions(units, **conv_net_params)

    # Up to the sun light
    for down_step in range(rabbit_hole_depth):
        units = tf.expand_dims(units, axis=2)
        units = tf.layers.conv2d_transpose(units, n_filters, filter_width, strides=(2, 1), padding='same')
        units = tf.squeeze(units, axis=2)

        # Skip connection
        units = units_for_skip_conn[-(down_step + 1)] + units

        units = stacked_convolutions(units, **conv_net_params)
    return units


def highway_convolutional_network(input_units,
                                  n_filters,
                                  filter_width=3,
                                  use_batch_norm=False,
                                  use_dilation=False,
                                  training_ph=None):
    if n_filters is None:
        # If number of filters is not given the number of filters
        # will be equal to the number of input features
        n_filters = input_units.get_shape().as_list()[-1]
    for n_layer, n_filt in enumerate(n_filters):
        if use_dilation:
            dilation_rate = 2**n_layer
        else:
            dilation_rate = 1
        units = tf.layers.conv1d(input_units,
                                 n_filt,
                                 filter_width,
                                 padding='same',
                                 dilation_rate=dilation_rate,
                                 kernel_initializer=xavier_initializer())
        if use_batch_norm:
            units = tf.layers.batch_normalization(units, training=training_ph)
        sigmoid_gate = tf.layers.dense(input_units, activation=tf.sigmoid, kernel_initializer=xavier_initializer())
        input_units = sigmoid_gate * input_units + (1 - sigmoid_gate) * units
        input_units = tf.nn.relu(input_units)
    return input_units


def embedding_layer(input_placeholder=None,
                    token_embedding_matrix=None,
                    n_tokens=None,
                    token_embedding_dim=None,
                    name=None,
                    trainable=True):
    if token_embedding_matrix is not None:
        tok_mat = token_embedding_matrix
        if trainable:
            Warning('Matrix of embeddings is passed to the embedding_layer, '
                    'possibly there is a pre-trained embedding matrix. '
                    'Embeddings paramenters are set to Trainable!')
    else:
        tok_mat = np.random.randn(n_tokens, token_embedding_dim).astype(np.float32) / np.sqrt(token_embedding_dim)
    tok_emb_mat = tf.Variable(tok_mat, name=name, trainable=trainable)
    embeddings = tf.nn.embedding_lookup(tok_emb_mat, input_placeholder)
    return embeddings


def character_embedding_network(char_placeholder, n_characters, char_embedding_dim, filter_width=7):
    char_emb_mat = np.random.randn(n_characters, char_embedding_dim).astype(np.float32) / np.sqrt(char_embedding_dim)
    char_emb_var = tf.Variable(char_emb_mat, trainable=True)
    with tf.variable_scope('Char_Emb_Network'):
        # Character embedding layer
        c_emb = tf.nn.embedding_lookup(char_emb_var, char_placeholder)

        # Character embedding network
        char_conv = tf.layers.conv2d(c_emb, char_embedding_dim, (1, filter_width), padding='same', name='char_conv')
        char_emb = tf.reduce_max(char_conv, axis=2)
    return char_emb


if __name__ == '__main__':
    batch_size = 4
    tokens = 16
    features = 50

    var = tf.Variable(np.random.randn(batch_size, tokens, features).astype(np.float32))
    u_shape(var, 2, 100, 2, 3)