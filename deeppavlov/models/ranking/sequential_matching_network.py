# Copyright 2018 Neural Networks and Deep Learning lab, MIPT
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
from typing import Optional

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.tf_base_matching_model import TensorflowBaseMatchingModel

log = getLogger(__name__)


@register('smn_nn')
class SMNNetwork(TensorflowBaseMatchingModel):
    """
    Tensorflow implementation of Sequential Matching Network

    Wu, Yu, et al. "Sequential Matching Network: A New Architecture for Multi-turn Response Selection in
    Retrieval-based Chatbots." ACL. 2017.
    https://arxiv.org/abs/1612.01627

    Based on authors' Tensorflow code: https://github.com/MarkWuNLP/MultiTurnResponseSelection

    Args:
        num_context_turns (int): A number of ``context`` turns in data samples.
        max_sequence_length (int): A maximum length of text sequences in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        learning_rate (float): Initial learning rate.
        emb_matrix (np.ndarray): An embeddings matrix to initialize an embeddings layer of a model.
        trainable_embeddings (bool): Whether train embeddings matrix or not.
        embedding_dim (int): Dimensionality of token (word) embeddings.
    """

    def __init__(self,
                 embedding_dim: int = 200,
                 max_sequence_length: int = 50,
                 learning_rate: float = 1e-3,
                 emb_matrix: Optional[np.ndarray] = None,
                 trainable_embeddings: bool = False,
                 *args,
                 **kwargs):

        self.max_sentence_len = max_sequence_length
        self.word_embedding_size = embedding_dim
        self.trainable = trainable_embeddings
        self.learning_rate = learning_rate
        self.emb_matrix = emb_matrix

        super(SMNNetwork, self).__init__(*args, **kwargs)

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)
        self._init_graph()
        self.sess.run(tf.global_variables_initializer())

        if self.load_path is not None:
            self.load()

    def _init_placeholders(self):
        with tf.variable_scope('inputs'):
            # Utterances and their lengths
            self.utterance_ph = tf.placeholder(tf.int32, shape=(None, self.num_context_turns, self.max_sentence_len))
            self.all_utterance_len_ph = tf.placeholder(tf.int32, shape=(None, self.num_context_turns))

            # Responses and their lengths
            self.response_ph = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
            self.response_len_ph = tf.placeholder(tf.int32, shape=(None,))

            # Labels
            self.y_true = tf.placeholder(tf.int32, shape=(None,))

    def _init_graph(self):
        self._init_placeholders()

        word_embeddings = tf.get_variable("word_embeddings_v",
                                          initializer=tf.constant(self.emb_matrix, dtype=tf.float32),
                                          trainable=self.trainable)

        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.utterance_ph)
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response_ph)
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.word_embedding_size, kernel_initializer=tf.orthogonal_initializer())
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.num_context_turns,
                                              axis=1)  # list of self.num_context_turns tensors with shape (?, 200)
        all_utterance_len = tf.unstack(self.all_utterance_len_ph, num=self.num_context_turns, axis=1)
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.word_embedding_size, self.word_embedding_size),
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.word_embedding_size, kernel_initializer=tf.orthogonal_initializer())
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU,
                                                       response_embeddings,
                                                       sequence_length=self.response_len_ph,
                                                       dtype=tf.float32,
                                                       scope='sentence_GRU')
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings)
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU,
                                                            utterance_embeddings,
                                                            sequence_length=utterance_len,
                                                            dtype=tf.float32,
                                                            scope='sentence_GRU')
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID',
                                          kernel_initializer=tf.contrib.keras.initializers.he_normal(),
                                          activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3),
                                                    padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              activation=tf.tanh, reuse=reuse,
                                              name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU,
                                           tf.stack(matching_vectors, axis=0, name='matching_stack'),
                                           # resulting shape: (10, ?, 50)
                                           dtype=tf.float32,
                                           time_major=True,
                                           scope='final_GRU')  # TODO: check time_major
        logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='final_v')
        self.y_pred = tf.nn.softmax(logits)
        self.logits = logits
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true, logits=logits))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        # Debug
        self.print_number_of_parameters()
