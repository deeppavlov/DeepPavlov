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

import shutil

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from deeppavlov.models.elmo.elmo_model import BidirectionalLanguageModel, weight_layers


def make_module_spec(options, weight_file):
    """Makes a module spec.

    Args:
      options: LM hyperparameters.
      weight_file: location of the hdf5 file with LM weights.

    Returns:
      A module spec object used for constructing a TF-Hub module.
    """

    def module_fn():
        """Spec function for a token embedding module."""
        # init
        _bos_id = 256
        _eos_id = 257
        _bow_id = 258
        _eow_id = 259
        _pad_id = 260

        _max_word_length = 50
        _parallel_iterations = 10
        _max_batch_size = 1024

        id_dtype = tf.int32
        id_nptype = np.int32
        max_word_length = tf.constant(_max_word_length, dtype=id_dtype, name='max_word_length')

        version = tf.constant('from_dp_1', dtype=tf.string, name='version')

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([_max_word_length], dtype=id_nptype)
            r[:] = _pad_id
            r[0] = _bow_id
            r[1] = c
            r[2] = _eow_id
            return tf.constant(r, dtype=id_dtype)

        bos_ids = _make_bos_eos(_bos_id)
        eos_ids = _make_bos_eos(_eos_id)

        def token2ids(token):
            with tf.name_scope("token2ids_preprocessor"):
                char_ids = tf.decode_raw(token, tf.uint8, name='decode_raw2get_char_ids')
                char_ids = tf.cast(char_ids, tf.int32, name='cast2int_token')
                char_ids = tf.strided_slice(char_ids, [0], [max_word_length - 2],
                                            [1], name='slice2resized_token')
                ids_num = tf.shape(char_ids)[0]
                fill_ids_num = (_max_word_length - 2) - ids_num
                pads = tf.fill([fill_ids_num], _pad_id)
                bow_token_eow_pads = tf.concat([[_bow_id], char_ids, [_eow_id], pads],
                                               0, name='concat2bow_token_eow_pads')
                return bow_token_eow_pads

        def sentence_tagging_and_padding(sen_dim):
            with tf.name_scope("sentence_tagging_and_padding_preprocessor"):
                sen = sen_dim[0]
                dim = sen_dim[1]
                extra_dim = tf.shape(sen)[0] - dim
                sen = tf.slice(sen, [0, 0], [dim, max_word_length], name='slice2sen')

                bos_sen_eos = tf.concat([[bos_ids], sen, [eos_ids]], 0, name='concat2bos_sen_eos')
                bos_sen_eos_plus_one = bos_sen_eos + 1
                bos_sen_eos_pads = tf.pad(bos_sen_eos_plus_one, [[0, extra_dim], [0, 0]],
                                          "CONSTANT", name='pad2bos_sen_eos_pads')
                return bos_sen_eos_pads

        # Input placeholders to the biLM.
        tokens = tf.placeholder(shape=(None, None), dtype=tf.string, name='ph2tokens')
        sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='ph2sequence_len')

        tok_shape = tf.shape(tokens)
        line_tokens = tf.reshape(tokens, shape=[-1], name='reshape2line_tokens')

        with tf.device('/cpu:0'):
            tok_ids = tf.map_fn(
                token2ids,
                line_tokens,
                dtype=tf.int32, back_prop=False, parallel_iterations=_parallel_iterations,
                name='map_fn2get_tok_ids')

        tok_ids = tf.reshape(tok_ids, [tok_shape[0], tok_shape[1], -1], name='reshape2tok_ids')
        with tf.device('/cpu:0'):
            sen_ids = tf.map_fn(
                sentence_tagging_and_padding,
                (tok_ids, sequence_len),
                dtype=tf.int32, back_prop=False, parallel_iterations=_parallel_iterations,
                name='map_fn2get_sen_ids')

        # Build the biLM graph.
        bilm = BidirectionalLanguageModel(options, str(weight_file),
                                          max_batch_size=_max_batch_size)

        embeddings_op = bilm(sen_ids)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        elmo_output = weight_layers('elmo_output', embeddings_op, l2_coef=0.0)

        weighted_op = elmo_output['weighted_op']
        mean_op = elmo_output['mean_op']
        word_emb = elmo_output['word_emb']
        lstm_outputs1 = elmo_output['lstm_outputs1']
        lstm_outputs2 = elmo_output['lstm_outputs2']

        hub.add_signature("tokens", {"tokens": tokens, "sequence_len": sequence_len},
                          {"elmo": weighted_op,
                           "default": mean_op,
                           "word_emb": word_emb,
                           "lstm_outputs1": lstm_outputs1,
                           "lstm_outputs2": lstm_outputs2,
                           "version": version})

        # #########################Next signature############################# #

        # Input placeholders to the biLM.
        def_strings = tf.placeholder(shape=(None), dtype=tf.string)
        def_tokens_sparse = tf.string_split(def_strings)
        def_tokens_dense = tf.sparse_to_dense(sparse_indices=def_tokens_sparse.indices,
                                              output_shape=def_tokens_sparse.dense_shape,
                                              sparse_values=def_tokens_sparse.values,
                                              default_value=''
                                              )
        def_mask = tf.not_equal(def_tokens_dense, '')
        def_int_mask = tf.cast(def_mask, dtype=tf.int32)
        def_sequence_len = tf.reduce_sum(def_int_mask, axis=-1)

        def_tok_shape = tf.shape(def_tokens_dense)
        def_line_tokens = tf.reshape(def_tokens_dense, shape=[-1], name='reshape2line_tokens')

        with tf.device('/cpu:0'):
            def_tok_ids = tf.map_fn(
                token2ids,
                def_line_tokens,
                dtype=tf.int32, back_prop=False, parallel_iterations=_parallel_iterations,
                name='map_fn2get_tok_ids')

        def_tok_ids = tf.reshape(def_tok_ids, [def_tok_shape[0], def_tok_shape[1], -1], name='reshape2tok_ids')
        with tf.device('/cpu:0'):
            def_sen_ids = tf.map_fn(
                sentence_tagging_and_padding,
                (def_tok_ids, def_sequence_len),
                dtype=tf.int32, back_prop=False, parallel_iterations=_parallel_iterations,
                name='map_fn2get_sen_ids')

        # Get ops to compute the LM embeddings.
        def_embeddings_op = bilm(def_sen_ids)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        def_elmo_output = weight_layers('elmo_output', def_embeddings_op, l2_coef=0.0, reuse=True)

        def_weighted_op = def_elmo_output['weighted_op']
        def_mean_op = def_elmo_output['mean_op']
        def_word_emb = def_elmo_output['word_emb']
        def_lstm_outputs1 = def_elmo_output['lstm_outputs1']
        def_lstm_outputs2 = def_elmo_output['lstm_outputs2']

        hub.add_signature("default", {"strings": def_strings},
                          {"elmo": def_weighted_op,
                           "default": def_mean_op,
                           "word_emb": def_word_emb,
                           "lstm_outputs1": def_lstm_outputs1,
                           "lstm_outputs2": def_lstm_outputs2,
                           "version": version})

    return hub.create_module_spec(module_fn)


def export2hub(weight_file, hub_dir, options):
    """Exports a TF-Hub module
    """

    spec = make_module_spec(options, str(weight_file))

    try:
        with tf.Graph().as_default():
            module = hub.Module(spec)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if hub_dir.exists():
                    shutil.rmtree(hub_dir)
                module.export(str(hub_dir), sess)
    finally:
        pass
