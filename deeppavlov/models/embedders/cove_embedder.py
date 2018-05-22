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

from overrides import overrides

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger
from opennmt.utils.cell import build_cell
from opennmt.layers import ConcatReducer
from typing import List

log = get_logger(__name__)


@register('cove')
class CoVeEmbedder(TFModel):
    def __init__(self, vocab_file_path: str, checkpoint_path: str, num_layers: int, cell_class_name: str,
                 num_units: int, dropout: int, residual_connections: bool, reduce_method: str, **kwargs):

        super().__init__(save_path=None, load_path=None, **kwargs)
        self.vocab_file_path = vocab_file_path
        self.checkpoint_path = checkpoint_path

        # TODO: reduce vectors (one vector per word in the input sentence) from encoder using this method
        self.reduce_method = reduce_method

        vocab = tf.contrib.lookup.index_table_from_file(vocabulary_file=self.vocab_file_path)
        self.tokens = tf.placeholder(tf.string, shape=(None, None), name='tokens')
        self.sequence_length = tf.placeholder(tf.int32, shape=[None, ], name='seq_length')
        indexes = vocab.lookup(self.tokens)

        checkpoint_values = self._load_checkpoint_values()
        embedd_mtx = tf.Variable(initial_value=checkpoint_values['seq2seq/encoder/w_embs'],
                                 trainable=False, dtype=tf.float32, name='w_embs')
        inputs = tf.nn.embedding_lookup(params=embedd_mtx, ids=indexes)

        if cell_class_name not in dir(tf.contrib.rnn):
            log.error("Provided cell_class_name hasn't been find into tf.contrib.rnn, try another name")
            exit(1)

        cell_class = getattr(tf.contrib.rnn, cell_class_name)

        with tf.variable_scope('seq2seq/encoder', reuse=tf.AUTO_REUSE):
            cell_fw = build_cell(
                num_layers,
                num_units,
                tf.estimator.ModeKeys.PREDICT,
                dropout=dropout,
                residual_connections=residual_connections,
                cell_class=cell_class)

            cell_bw = build_cell(
                num_layers,
                num_units,
                tf.estimator.ModeKeys.PREDICT,
                dropout=dropout,
                residual_connections=residual_connections,
                cell_class=cell_class)

            encoder_outputs_tup, encoder_state_tup = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                inputs,
                sequence_length=self.sequence_length,
                dtype=inputs.dtype)

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            tf_vars = []
            tf_names = []
            for name, value in checkpoint_values.items():
                if 'bidirectional_rnn' in name:
                    tf_vars.append(tf.get_variable(name))
                    tf_names.append(name)
                    print(name, tf.get_variable(name).shape, checkpoint_values[name].shape)
            placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
            assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

        reducer = ConcatReducer()

        self.encoder_outputs = reducer.zip_and_reduce(encoder_outputs_tup[0], encoder_outputs_tup[1])
        self.encoder_state = reducer.zip_and_reduce(encoder_state_tup[0], encoder_state_tup[1])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())
        self.sess.run(assign_ops, {p: checkpoint_values[name] for p, name in zip(placeholders, tf_names)})

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

    def _load_checkpoint_values(self, *args, **kwargs):
        """
        Load weights from checkpoint
        """

        # loading checkpoint values:
        var_list = tf.train.list_variables(self.checkpoint_path)
        values = {}
        for name, shape in var_list:
            if not name.startswith("global_step"):
                values[name] = np.zeros(shape)
        reader = tf.train.load_checkpoint(self.checkpoint_path)
        for name in values:
            values[name] += reader.get_tensor(name)
        return values

    @overrides
    def __call__(self, batch, lengths, *args, **kwargs):
        """
        Embed data
        """
        return self._encode(batch, lengths)

    def _encode(self, tokens: List[str], lengths: List[int]):

        # TODO: add concatenation or smth like this
        lengths = self.get_length
        embedded_tokens = self.sess.run(self.encoder_state, {self.sequence_length: lengths, self.tokens: tokens})

        return embedded_tokens
