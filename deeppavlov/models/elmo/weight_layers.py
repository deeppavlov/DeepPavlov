# originally based on https://github.com/allenai/bilm-tf/blob/master/bilm/elmo.py

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

import tensorflow as tf


def weight_layers(name, bilm_ops, l2_coef=None,
                  use_top_only=False, do_layer_norm=False, reuse = False):
    '''
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.

    For each output layer, this returns two ops.  The first computes
        a layer specific weighted average of the biLM layers, and
        the second the l2 regularizer loss term.
    The regularization terms are also add to tf.GraphKeys.REGULARIZATION_LOSSES

    Input:
        name = a string prefix used for the trainable variable names
        bilm_ops = the tensorflow ops returned to compute internal
            representations from a biLM.  This is the return value
            from BidirectionalLanguageModel(...)(ids_placeholder)
        l2_coef: the l2 regularization coefficient $\lambda$.
            Pass None or 0.0 for no regularization.
        use_top_only: if True, then only use the top layer.
        do_layer_norm: if True, then apply layer normalization to each biLM
            layer before normalizing
        reuse: reuse an aggregation variable scope.

    Output:
        {
            'weighted_op': op to compute weighted average for output,
            'regularization_op': op to compute regularization term
        }
    '''
    def _l2_regularizer(weights):
        if l2_coef is not None:
            return l2_coef * tf.reduce_sum(tf.square(weights))
        else:
            return 0.0

    # Get ops for computing LM embeddings and mask
    lm_embeddings = bilm_ops['lm_embeddings']
    mask = bilm_ops['mask']

    n_lm_layers = int(lm_embeddings.get_shape()[1])
    lm_dim = int(lm_embeddings.get_shape()[3])
    # import pdb; pdb.set_trace()

    with tf.control_dependencies([lm_embeddings, mask]):
        # Cast the mask and broadcast for layer use.
        mask_float = tf.cast(mask, 'float32')
        broadcast_mask = tf.expand_dims(mask_float, axis=-1)

        def _do_ln(x):
            # do layer normalization excluding the mask
            x_masked = x * broadcast_mask
            N = tf.reduce_sum(mask_float) * lm_dim
            mean = tf.reduce_sum(x_masked) / N
            variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask)**2) / N
            return tf.nn.batch_normalization(
                x, mean, variance, None, None, 1E-12
            )

        if use_top_only:
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
            # just the top layer
            sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
            # no regularization
            reg = 0.0
        else:
            with tf.variable_scope("aggregation", reuse = reuse):
                W = tf.get_variable(
                    '{}_ELMo_W'.format(name),
                    shape=(n_lm_layers, ),
                    initializer=tf.zeros_initializer,
                    regularizer=_l2_regularizer,
                    trainable=True,
                )

            # normalize the weights
            normed_weights = tf.split(
                tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers
            )
            # split LM layers
            layers = tf.split(lm_embeddings, n_lm_layers, axis=1)

            # compute the weighted, normalized LM activations
            pieces = []
            for w, t in zip(normed_weights, layers):
                if do_layer_norm:
                    pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                else:
                    pieces.append(w * tf.squeeze(t, squeeze_dims=1))
            sum_pieces = tf.add_n(pieces)

            # get the regularizer
            reg = [
                r for r in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                if r.name.find('{}_ELMo_W/'.format(name)) >= 0
            ]
            if len(reg) != 1:
                raise ValueError

        # scale the weighted sum by gamma

        with tf.variable_scope("aggregation", reuse = reuse):
            gamma = tf.get_variable(
                '{}_ELMo_gamma'.format(name),
                shape=(1, ),
                initializer=tf.ones_initializer,
                regularizer=None,
                trainable=True,
            )

        weighted_lm_layers = sum_pieces * gamma
        weighted_lm_layers_masked = sum_pieces * broadcast_mask

        weighted_lm_layers_sum = tf.reduce_sum(weighted_lm_layers_masked, 1)

        mask_sum = tf.reduce_sum(mask_float, 1)
        mask_sum = tf.maximum(mask_sum, [1])

        weighted_lm_layers_mean = weighted_lm_layers_sum / tf.expand_dims(mask_sum, -1)

        word_emb_2n = tf.squeeze(layers[0], [1])
        word_emb_1n = tf.slice(word_emb_2n, [0, 0, 0], [-1, -1, lm_dim // 2])  # to 512
        lstm_outputs1 = tf.squeeze(layers[1], [1])
        lstm_outputs2 = tf.squeeze(layers[2], [1])

        ret = {'weighted_op': weighted_lm_layers,
               'mean_op': weighted_lm_layers_mean,
               'regularization_op': reg,
               'word_emb': word_emb_1n,
               'lstm_outputs1': lstm_outputs1,
               'lstm_outputs2': lstm_outputs2, }

    return ret
