# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
from typing import List, Union, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.contrib.layers import xavier_initializer

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.layers.tf_layers import bi_rnn
from deeppavlov.models.bert.bert_sequence_tagger import BertSequenceNetwork, token_from_subtoken

log = getLogger(__name__)


def gather_indexes(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    """
    Args:
        A: a tensor with data
        B: an integer tensor with indexes

    Returns:
        `answer` a tensor such that ``answer[i, j] = A[i, B[i, j]]``.
        In case `B` is one-dimensional, the output is ``answer[i] = A[i, B[i]]``

    """
    are_indexes_one_dim = (kb.ndim(B) == 1)
    if are_indexes_one_dim:
        B = tf.expand_dims(B, -1)
    first_dim_indexes = tf.expand_dims(tf.range(tf.shape(B)[0]), -1)
    first_dim_indexes = tf.tile(first_dim_indexes, [1, tf.shape(B)[1]])
    indexes = tf.stack([first_dim_indexes, B], axis=-1)
    answer = tf.gather_nd(A, indexes)
    if are_indexes_one_dim:
        answer = answer[:,0]
    return answer


def biaffine_layer(deps: tf.Tensor, heads: tf.Tensor, deps_dim: int,
                   heads_dim: int, output_dim: int, name: str = "biaffine_layer") -> tf.Tensor:
    """Implements a biaffine layer from [Dozat, Manning, 2016].

    Args:
        deps: the 3D-tensor of dependency states,
        heads: the 3D-tensor of head states,
        deps_dim: the dimension of dependency states,
        heads_dim: the dimension of head_states,
        output_dim: the output dimension
        name: the name of a layer

    Returns:
        `answer` the output 3D-tensor

    """
    input_shape = [kb.shape(deps)[i] for i in range(tf.keras.backend.ndim(deps))]
    first_input = tf.reshape(deps, [-1, deps_dim])  # first_input.shape = (B*L, D1)
    second_input = tf.reshape(heads, [-1, heads_dim])  # second_input.shape = (B*L, D2)
    with tf.variable_scope(name):
        kernel_shape = (deps_dim, heads_dim * output_dim)
        kernel = tf.get_variable('kernel', shape=kernel_shape, initializer=xavier_initializer())
        first = tf.matmul(first_input, kernel)  # (B*L, D2*H)
        first = tf.reshape(first, [-1, heads_dim, output_dim])  # (B*L, D2, H)
        answer = kb.batch_dot(first, second_input, axes=[1, 1])  # (B*L, H)
        first_bias = tf.get_variable('first_bias', shape=(deps_dim, output_dim),
                                     initializer=xavier_initializer())
        answer += tf.matmul(first_input, first_bias)
        second_bias = tf.get_variable('second_bias', shape=(heads_dim, output_dim),
                                      initializer=xavier_initializer())
        answer += tf.matmul(second_input, second_bias)
        label_bias = tf.get_variable('label_bias', shape=(output_dim,),
                                     initializer=xavier_initializer())
        answer = kb.bias_add(answer, label_bias)
        answer = tf.reshape(answer, input_shape[:-1] + [output_dim])  # (B, L, H)
    return answer


def biaffine_attention(deps: tf.Tensor, heads: tf.Tensor, name="biaffine_attention") -> tf.Tensor:
    """Implements a trainable matching layer between two families of embeddings.

    Args:
        deps: the 3D-tensor of dependency states,
        heads: the 3D-tensor of head states,
        name: the name of a layer

    Returns:
        `answer` a 3D-tensor of pairwise scores between deps and heads

    """
    deps_dim_int = deps.get_shape().as_list()[-1]
    heads_dim_int = heads.get_shape().as_list()[-1]
    assert deps_dim_int == heads_dim_int
    with tf.variable_scope(name):
        kernel_shape = (deps_dim_int, heads_dim_int)
        kernel = tf.get_variable('kernel', shape=kernel_shape, initializer=tf.initializers.identity())
        first_bias = tf.get_variable('first_bias', shape=(kernel_shape[0], 1),
                                     initializer=xavier_initializer())
        second_bias = tf.get_variable('second_bias', shape=(kernel_shape[1], 1),
                                      initializer=xavier_initializer())
        # deps.shape = (B, L, D)
        # first.shape = (B, L, D), first_rie = sum_d deps_{rid} kernel_{de}
        first = tf.tensordot(deps, kernel, axes=[-1, -2])
        answer = tf.matmul(first, heads, transpose_b=True)  # answer.shape = (B, L, L)
        # add bias over x axis
        first_bias_term = tf.tensordot(deps, first_bias, axes=[-1, -2])
        answer += first_bias_term
        # add bias over y axis
        second_bias_term = tf.tensordot(heads, second_bias, axes=[-1, -2])  # (B, L, 1)
        second_bias_term = tf.transpose(second_bias_term, [0, 2, 1])  # (B, 1, L)
        answer += second_bias_term
    return answer


@register('bert_syntax_parser')
class BertSyntaxParser(BertSequenceNetwork):
    """BERT-based model for syntax parsing.
    For each word the model predicts the index of its syntactic head
    and the label of the dependency between this head and the current word.
    See :class:`deeppavlov.models.bert.bert_sequence_tagger.BertSequenceNetwork`
    for the description of inherited parameters.

    Args:
        n_deps: number of distinct syntactic dependencies
        embeddings_dropout: dropout for embeddings in biaffine layer
        state_size: the size of hidden state in biaffine layer
        dep_state_size: the size of hidden state in biaffine layer
        use_birnn: whether to use bidirection rnn after BERT layers.
            Set it to `True` as it leads to much higher performance at least on large datasets
        birnn_cell_type: the type of Bidirectional RNN. Either `lstm` or `gru`
        birnn_hidden_size: number of hidden units in the BiRNN layer in each direction
        return_probas: set this to `True` if you need the probabilities instead of raw answers
        predict tags: whether to predict morphological tags together with syntactic information
        n_tags: the number of morphological tags
        tag_weight: the weight of tag model loss in multitask training
    """

    def __init__(self,
                 n_deps: int,
                 keep_prob: float,
                 bert_config_file: str,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 embeddings_dropout: float = 0.0,
                 encoder_layer_ids: List[int] = (-1,),
                 encoder_dropout: float = 0.0,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 state_size: int = 256,
                 use_birnn: bool = True,
                 birnn_cell_type: str = 'lstm',
                 birnn_hidden_size: int = 256,
                 ema_decay: float = None,
                 ema_variables_on_cpu: bool = True,
                 predict_tags = False,
                 n_tags = None,
                 tag_weight = 1.0,
                 return_probas: bool = False,
                 freeze_embeddings: bool = False,
                 learning_rate: float = 1e-3,
                 bert_learning_rate: float = 2e-5,
                 min_learning_rate: float = 1e-07,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: float = 1.0,
                 **kwargs) -> None:
        self.n_deps = n_deps
        self.embeddings_dropout = embeddings_dropout
        self.state_size = state_size
        self.use_birnn = use_birnn
        self.birnn_cell_type = birnn_cell_type
        self.birnn_hidden_size = birnn_hidden_size
        self.return_probas = return_probas
        self.predict_tags = predict_tags
        self.n_tags = n_tags
        self.tag_weight = tag_weight
        if self.predict_tags and self.n_tags is None:
            raise ValueError("n_tags should be given if `predict_tags`=True.")
        super().__init__(keep_prob=keep_prob,
                         bert_config_file=bert_config_file,
                         pretrained_bert=pretrained_bert,
                         attention_probs_keep_prob=attention_probs_keep_prob,
                         hidden_keep_prob=hidden_keep_prob,
                         encoder_layer_ids=encoder_layer_ids,
                         encoder_dropout=encoder_dropout,
                         optimizer=optimizer,
                         weight_decay_rate=weight_decay_rate,
                         ema_decay=ema_decay,
                         ema_variables_on_cpu=ema_variables_on_cpu,
                         freeze_embeddings=freeze_embeddings,
                         learning_rate=learning_rate,
                         bert_learning_rate=bert_learning_rate,
                         min_learning_rate=min_learning_rate,
                         learning_rate_drop_div=learning_rate_drop_div,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         load_before_drop=load_before_drop,
                         clip_norm=clip_norm,
                         **kwargs)

    def _init_graph(self) -> None:
        self._init_placeholders()

        units = super()._init_graph()

        with tf.variable_scope('ner'):
            units = token_from_subtoken(units, self.y_masks_ph)
            if self.use_birnn:
                units, _ = bi_rnn(units,
                                  self.birnn_hidden_size,
                                  cell_type=self.birnn_cell_type,
                                  seq_lengths=self.seq_lengths,
                                  name='birnn')
                units = tf.concat(units, -1)
            # for heads
            head_embeddings = tf.layers.dense(units, units=self.state_size, activation="relu")
            head_embeddings = tf.nn.dropout(head_embeddings, self.embeddings_keep_prob_ph)
            dep_embeddings = tf.layers.dense(units, units=self.state_size, activation="relu")
            dep_embeddings = tf.nn.dropout(dep_embeddings, self.embeddings_keep_prob_ph)
            self.dep_head_similarities = biaffine_attention(dep_embeddings, head_embeddings)
            self.dep_heads = tf.argmax(self.dep_head_similarities, -1)
            self.dep_head_probs = tf.nn.softmax(self.dep_head_similarities)
            # for dependency types
            head_embeddings = tf.layers.dense(units, units=self.state_size, activation="relu")
            head_embeddings = tf.nn.dropout(head_embeddings, self.embeddings_keep_prob_ph)
            dep_embeddings = tf.layers.dense(units, units=self.state_size, activation="relu")
            dep_embeddings = tf.nn.dropout(dep_embeddings, self.embeddings_keep_prob_ph)
            # matching each word with its head
            head_embeddings = gather_indexes(head_embeddings, self.y_head_ph)
            self.dep_logits = biaffine_layer(dep_embeddings, head_embeddings, 
                                             deps_dim=self.state_size, heads_dim=self.state_size, 
                                             output_dim=self.n_deps)
            self.deps = tf.argmax(self.dep_logits, -1)
            self.dep_probs = tf.nn.softmax(self.dep_logits)
            if self.predict_tags:
                tag_embeddings = tf.layers.dense(units, units=self.state_size, activation="relu")
                tag_embeddings = tf.nn.dropout(tag_embeddings, self.embeddings_keep_prob_ph)
                self.tag_logits = tf.layers.dense(tag_embeddings, units=self.n_tags)
                self.tags = tf.argmax(self.tag_logits, -1)
                self.tag_probs = tf.nn.softmax(self.tag_logits)
        with tf.variable_scope("loss"):
            tag_mask = self._get_tag_mask()
            y_mask = tf.cast(tag_mask, tf.float32)
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_head_ph,
                                                               logits=self.dep_head_similarities,
                                                               weights=y_mask)
            self.loss += tf.losses.sparse_softmax_cross_entropy(labels=self.y_dep_ph,
                                                                logits=self.dep_logits,
                                                                weights=y_mask)
            if self.predict_tags:
                tag_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_tag_ph,
                                                                  logits=self.tag_logits,
                                                                  weights=y_mask)
                self.loss += self.tag_weight_ph * tag_loss

    def _init_placeholders(self) -> None:
        super()._init_placeholders()
        self.y_head_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_head_ph')
        self.y_dep_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_dep_ph')
        if self.predict_tags:
            self.y_tag_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_tag_ph')
        self.y_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_mask_ph')
        self.embeddings_keep_prob_ph = tf.placeholder_with_default(
            1.0, shape=[], name="embeddings_keep_prob_ph")
        if self.predict_tags:
            self.tag_weight_ph = tf.placeholder_with_default(1.0, shape=[], name="tag_weight_ph")

    def _build_feed_dict(self, input_ids, input_masks, y_masks, 
                         y_head=None, y_dep=None, y_tag=None) -> dict:
        y_masks = np.concatenate([np.ones_like(y_masks[:,:1]), y_masks[:, 1:]], axis=1)
        feed_dict = self._build_basic_feed_dict(input_ids, input_masks, train=(y_head is not None))
        feed_dict[self.y_masks_ph] = y_masks
        if y_head is not None:
            y_head = zero_pad(y_head)
            y_head = np.concatenate([np.zeros_like(y_head[:,:1]), y_head], axis=1)
            y_dep = zero_pad(y_dep)
            y_dep = np.concatenate([np.zeros_like(y_dep[:,:1]), y_dep], axis=1)
            feed_dict.update({self.embeddings_keep_prob_ph: 1.0 - self.embeddings_dropout,
                              self.y_head_ph: y_head,
                              self.y_dep_ph: y_dep})
            if self.predict_tags:
                y_tag = np.concatenate([np.zeros_like(y_tag[:,:1]), y_tag], axis=1)
                feed_dict.update({self.y_tag_ph: y_tag, self.tag_weight_ph: self.tag_weight})
        return feed_dict

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray]) \
            -> Union[Tuple[List[Union[List[int], np.ndarray]], List[List[int]]],
                     Tuple[List[Union[List[int], np.ndarray]], List[List[int]], List[List[int]]]]:

        """ Predicts the outputs for a batch of inputs.
        By default (``return_probas`` = `False` and ``predict_tags`` = `False`) it returns two output batches.
        The first is the batch of head indexes: `i` stands for `i`-th word in the sequence,
        where numeration starts with 1. `0` is predicted for the syntactic root of the sentence.
        The second is the batch of indexes for syntactic dependencies.
        In case ``return_probas`` = `True` we return the probability distribution over possible heads
        instead of the position of the most probable head. For a sentence of length `k` the output
        is an array of shape `k * (k+1)`.
        In case ``predict_tags`` = `True` the model additionally returns the index of the most probable
        morphological tag for each word. The batch of such indexes becomes the third output of the function.

        Returns:
            `pred_heads_to_return`, either a batch of most probable head positions for each token
            (in case ``return_probas`` = `False`)
            or a batch of probability distribution over token head positions

            `pred_deps`, the indexes of token dependency relations

            `pred_tags`: the indexes of token morphological tags (only if ``predict_tags`` = `True`)

        """
        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks)
        if self.ema:
            self.sess.run(self.ema.switch_to_test_op)
        if self.return_probas:
            pred_head_probs, pred_heads, seq_lengths =\
                 self.sess.run([self.dep_head_probs, self.dep_heads, self.seq_lengths], feed_dict=feed_dict)
            pred_heads_to_return = [np.array(p[1:l,:l]) for l, p in zip(seq_lengths, pred_head_probs)]
        else:
            pred_heads, seq_lengths = self.sess.run([self.dep_heads, self.seq_lengths], feed_dict=feed_dict)
            pred_heads_to_return = [p[1:l] for l, p in zip(seq_lengths, pred_heads)]
        feed_dict[self.y_head_ph] = pred_heads
        pred_deps = self.sess.run(self.deps, feed_dict=feed_dict)
        pred_deps = [p[1:l] for l, p in zip(seq_lengths, pred_deps)]
        answer = [pred_heads_to_return, pred_deps]
        if self.predict_tags:
            pred_tags = self.sess.run(self.tags, feed_dict=feed_dict)
            pred_tags = [p[1:l] for l, p in zip(seq_lengths, pred_tags)]
            answer.append(pred_tags)
        return tuple(answer)
