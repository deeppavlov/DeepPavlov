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
from typing import List, Any, Tuple, Union, Dict

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import xavier_initializer
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.models.bert.bert_sequence_tagger import BertSequenceNetwork, token_from_subtoken,\
    ExponentialMovingAverage
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.core.layers.tf_layers import bi_rnn
from deeppavlov.core.models.tf_model import LRScheduledTFModel

log = getLogger(__name__)


def biaffine_attention(deps, heads, name="biaffine_attention"):
    deps_dim_int = deps.get_shape().as_list()[-1]
    heads_dim_int = heads.get_shape().as_list()[-1]
    assert deps_dim_int == heads_dim_int
    with tf.variable_scope(name):
        kernel_shape = (deps_dim_int, deps_dim_int)
        kernel = tf.get_variable('kernel', shape=kernel_shape, initializer=tf.initializers.identity())
        first_bias = tf.get_variable('first_bias', shape=(kernel_shape[0], 1),
                                     initializer=xavier_initializer())
        second_bias = tf.get_variable('second_bias', shape=(kernel_shape[1], 1),
                                      initializer=xavier_initializer())
        # deps.shape = (B, L, D)
        first = tf.tensordot(deps, kernel, axes=[-1,-2])  # first.shape = (B, L, D), first_rie = sum_d x_{rid} a_{de}
        answer = tf.matmul(first, heads, transpose_b=True)  # answer.shape = (B, L, L)
        # add bias over x axis
        first_bias_term = tf.tensordot(deps, first_bias, axes=[-1,-2])
        answer += first_bias_term
        # add bias over y axis
        second_bias_term = tf.tensordot(heads, second_bias, axes=[-1,-2]) # (B, L, 1)
        second_bias_term = tf.transpose(second_bias_term, [0, 2, 1])  # (B, 1, L)
        answer += second_bias_term
    return answer


@register('bert_syntax_parser')
class BertSyntaxParser(BertSequenceNetwork):
    """BERT-based model for syntax parsing.

    Args:
        n_deps: number of distinct syntactic dependencies
        keep_prob: dropout keep_prob for non-Bert layers
        bert_config_file: path to Bert configuration file
        pretrained_bert: pretrained Bert checkpoint
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        use_chl_decoding: whether to use Chu-Liu-Edmonds decoding
        encoder_layer_ids: list of averaged layers from Bert encoder (layer ids)
            optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
            weight_decay_rate: L2 weight decay for `AdamWeightDecayOptimizer`
        ema_decay: what exponential moving averaging to use for network parameters, value from 0.0 to 1.0.
            Values closer to 1.0 put weight on the parameters history and values closer to 0.0 corresponds put weight
            on the current parameters.
        ema_variables_on_cpu: whether to put EMA variables to CPU. It may save a lot of GPU memory
        return_probas: set True if return class probabilites instead of most probable label needed
        freeze_embeddings: set True to not train input embeddings set True to
            not train input embeddings set True to not train input embeddings
        learning_rate: learning rate of the NER head
        bert_learning_rate: learning rate of the BERT body
            min_learning_rate: min value of learning rate if learning rate decay is used
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        clip_norm: clip gradients by norm
    """

    def __init__(self,
                 n_deps: int,
                 state_size: int,
                 keep_prob: float,
                 bert_config_file: str,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 embeddings_dropout: float = 0.0,
                 use_chl_decoding: bool = False,
                 encoder_layer_ids: List[int] = (-1,),
                 encoder_dropout: float = 0.0,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 ema_decay: float = None,
                 ema_variables_on_cpu: bool = True,
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
        self.state_size = state_size
        self.embeddings_dropout = embeddings_dropout
        self.use_chl_decoding = use_chl_decoding
        self.return_probas = return_probas
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
            y_masks_with_cls_ph = tf.concat(tf.ones_like(self.y_masks_ph[:,:1]), self.y_masks_ph[:,1:], axis=1)
            units = token_from_subtoken(units, y_masks_with_cls_ph)
            head_embeddings = tf.layers.dense(units, units=self.state_size, activation="relu")
            head_embeddings = tf.nn.dropout(head_embeddings, self.embeddings_keep_prob_ph)
            dep_embeddings = tf.layers.dense(units, units=self.state_size, activation="relu")
            dep_embeddings = tf.nn.dropout(dep_embeddings, self.embeddings_keep_prob_ph)
            self.dep_head_similarities = biaffine_attention(dep_embeddings, head_embeddings)
            self.dep_heads = tf.argmax(self.dep_head_similarities, -1)
            self.dep_head_probs = tf.nn.softmax(self.dep_head_similarities)

        with tf.variable_scope("loss"):
            tag_mask = self._get_tag_mask()
            y_mask = tf.cast(tag_mask, tf.float32)
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y_dep_ph,
                                                               logits=self.dep_head_similarities,
                                                               weights=y_mask)

    def _init_placeholders(self) -> None:
        super()._init_placeholders()
        self.y_dep_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_dep_ph')
        self.y_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_mask_ph')
        self.embeddings_keep_prob_ph = tf.placeholder_with_default(
            1.0, shape=[], name="embeddings_keep_prob_ph")


    def _build_feed_dict(self, input_ids, input_masks, y_masks, token_types=None,
                         y_dep=None):
        feed_dict = self._build_basic_feed_dict(
            input_ids, input_masks, token_types=token_types, train=(y_dep is not None))
        feed_dict[self.y_masks_ph] = y_masks
        if y_dep is not None:
            feed_dict[self.embeddings_keep_prob_ph] = 1.0 - self.embeddings_dropout,
            feed_dict[self.y_dep_ph] = y_dep
        return feed_dict

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray]) -> Union[List[List[int]], List[np.ndarray]]:
        """ Predicts tag indices for a given subword tokens batch

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            Predictions indices or predicted probabilities fro each token (not subtoken)

        """
        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks)
        if self.ema:
            self.sess.run(self.ema.switch_to_test_op)
        pred, seq_lengths = self.sess.run([self.y_dep_heads, self.seq_lengths], feed_dict=feed_dict)
        pred = [p[:l] for l, p in zip(seq_lengths, pred)]
        return pred
