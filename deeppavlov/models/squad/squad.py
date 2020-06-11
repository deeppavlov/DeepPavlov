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
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.check_gpu import check_gpu_existence
from deeppavlov.core.common.registry import register
from deeppavlov.core.layers.tf_layers import cudnn_bi_gru, variational_dropout
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.squad.utils import dot_attention, simple_attention, PtrNet, CudnnGRU, CudnnCompatibleGRU

logger = getLogger(__name__)


@register('squad_model')
class SquadModel(LRScheduledTFModel):
    """
    SquadModel predicts answer start and end position in given context by given question.

    High level architecture:
    Word embeddings -> Contextual embeddings -> Question-Context Attention -> Self-attention -> Pointer Network

    If noans_token flag is True, then special noans_token is added to output of self-attention layer.
    Pointer Network can select noans_token if there is no answer in given context.

    Parameters:
        word_emb: pretrained word embeddings
        char_emb: pretrained char embeddings
        context_limit: max context length in tokens
        question_limit: max question length in tokens
        char_limit: max number of characters in token
        char_hidden_size: hidden size of charRNN
        encoder_hidden_size: hidden size of encoder RNN
        attention_hidden_size: size of projection layer in attention
        keep_prob: dropout keep probability
        min_learning_rate: minimal learning rate, is used in learning rate decay
        noans_token: boolean, flags whether to use special no_ans token to make model able not to answer on question
    """

    def __init__(self, word_emb: np.ndarray, char_emb: np.ndarray, context_limit: int = 450, question_limit: int = 150,
                 char_limit: int = 16, train_char_emb: bool = True, char_hidden_size: int = 100,
                 encoder_hidden_size: int = 75, attention_hidden_size: int = 75, keep_prob: float = 0.7,
                 min_learning_rate: float = 0.001, noans_token: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)

        self.init_word_emb = word_emb
        self.init_char_emb = char_emb
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.char_limit = char_limit
        self.train_char_emb = train_char_emb
        self.char_hidden_size = char_hidden_size
        self.hidden_size = encoder_hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.keep_prob = keep_prob
        self.min_learning_rate = min_learning_rate
        self.noans_token = noans_token

        self.word_emb_dim = self.init_word_emb.shape[1]
        self.char_emb_dim = self.init_char_emb.shape[1]

        self.last_impatience = 0
        self.lr_impatience = 0

        if check_gpu_existence():
            self.GRU = CudnnGRU
        else:
            self.GRU = CudnnCompatibleGRU

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_graph()

        self._init_optimizer()

        self.sess.run(tf.global_variables_initializer())

        # Try to load the model (if there are some model files the model will be loaded from them)
        if self.load_path is not None:
            self.load()

    def _init_graph(self):
        self._init_placeholders()

        self.word_emb = tf.get_variable("word_emb", initializer=tf.constant(self.init_word_emb, dtype=tf.float32),
                                        trainable=False)
        self.char_emb = tf.get_variable("char_emb", initializer=tf.constant(self.init_char_emb, dtype=tf.float32),
                                        trainable=self.train_char_emb)

        self.c_mask = tf.cast(self.c_ph, tf.bool)
        self.q_mask = tf.cast(self.q_ph, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        bs = tf.shape(self.c_ph)[0]
        self.c_maxlen = tf.reduce_max(self.c_len)
        self.q_maxlen = tf.reduce_max(self.q_len)
        self.c = tf.slice(self.c_ph, [0, 0], [bs, self.c_maxlen])
        self.q = tf.slice(self.q_ph, [0, 0], [bs, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [bs, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [bs, self.q_maxlen])
        self.cc = tf.slice(self.cc_ph, [0, 0, 0], [bs, self.c_maxlen, self.char_limit])
        self.qc = tf.slice(self.qc_ph, [0, 0, 0], [bs, self.q_maxlen, self.char_limit])
        self.cc_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.cc, tf.bool), tf.int32), axis=2), [-1])
        self.qc_len = tf.reshape(tf.reduce_sum(tf.cast(tf.cast(self.qc, tf.bool), tf.int32), axis=2), [-1])
        # to remove char sequences with len equal zero (padded tokens)
        self.cc_len = tf.maximum(tf.ones_like(self.cc_len), self.cc_len)
        self.qc_len = tf.maximum(tf.ones_like(self.qc_len), self.qc_len)
        self.y1 = tf.one_hot(self.y1_ph, depth=self.context_limit)
        self.y2 = tf.one_hot(self.y2_ph, depth=self.context_limit)
        self.y1 = tf.slice(self.y1, [0, 0], [bs, self.c_maxlen])
        self.y2 = tf.slice(self.y2, [0, 0], [bs, self.c_maxlen])

        if self.noans_token:
            # we use additional 'no answer' token to allow model not to answer on question
            # later we will add 'no answer' token as first token in context question-aware representation
            self.y1 = tf.one_hot(self.y1_ph, depth=self.context_limit + 1)
            self.y2 = tf.one_hot(self.y2_ph, depth=self.context_limit + 1)
            self.y1 = tf.slice(self.y1, [0, 0], [bs, self.c_maxlen + 1])
            self.y2 = tf.slice(self.y2, [0, 0], [bs, self.c_maxlen + 1])

        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                cc_emb = tf.reshape(tf.nn.embedding_lookup(self.char_emb, self.cc),
                                    [bs * self.c_maxlen, self.char_limit, self.char_emb_dim])
                qc_emb = tf.reshape(tf.nn.embedding_lookup(self.char_emb, self.qc),
                                    [bs * self.q_maxlen, self.char_limit, self.char_emb_dim])

                cc_emb = variational_dropout(cc_emb, keep_prob=self.keep_prob_ph)
                qc_emb = variational_dropout(qc_emb, keep_prob=self.keep_prob_ph)

                _, (state_fw, state_bw) = cudnn_bi_gru(cc_emb, self.char_hidden_size, seq_lengths=self.cc_len,
                                                       trainable_initial_states=True)
                cc_emb = tf.concat([state_fw, state_bw], axis=1)

                _, (state_fw, state_bw) = cudnn_bi_gru(qc_emb, self.char_hidden_size, seq_lengths=self.qc_len,
                                                       trainable_initial_states=True,
                                                       reuse=True)
                qc_emb = tf.concat([state_fw, state_bw], axis=1)

                cc_emb = tf.reshape(cc_emb, [bs, self.c_maxlen, 2 * self.char_hidden_size])
                qc_emb = tf.reshape(qc_emb, [bs, self.q_maxlen, 2 * self.char_hidden_size])

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_emb, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_emb, self.q)

            c_emb = tf.concat([c_emb, cc_emb], axis=2)
            q_emb = tf.concat([q_emb, qc_emb], axis=2)

        with tf.variable_scope("encoding"):
            rnn = self.GRU(num_layers=3, num_units=self.hidden_size, batch_size=bs,
                           input_size=c_emb.get_shape().as_list()[-1],
                           keep_prob=self.keep_prob_ph)
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, att_size=self.attention_hidden_size,
                                   keep_prob=self.keep_prob_ph)
            rnn = self.GRU(num_layers=1, num_units=self.hidden_size, batch_size=bs,
                           input_size=qc_att.get_shape().as_list()[-1], keep_prob=self.keep_prob_ph)
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(att, att, mask=self.c_mask, att_size=self.attention_hidden_size,
                                     keep_prob=self.keep_prob_ph)
            rnn = self.GRU(num_layers=1, num_units=self.hidden_size, batch_size=bs,
                           input_size=self_att.get_shape().as_list()[-1], keep_prob=self.keep_prob_ph)
            match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("pointer"):
            init = simple_attention(q, self.hidden_size, mask=self.q_mask, keep_prob=self.keep_prob_ph)
            pointer = PtrNet(cell_size=init.get_shape().as_list()[-1], keep_prob=self.keep_prob_ph)
            if self.noans_token:
                noans_token = tf.Variable(tf.random_uniform((match.get_shape().as_list()[-1],), -0.1, 0.1), tf.float32)
                noans_token = tf.nn.dropout(noans_token, keep_prob=self.keep_prob_ph)
                noans_token = tf.expand_dims(tf.tile(tf.expand_dims(noans_token, axis=0), [bs, 1]), axis=1)
                match = tf.concat([noans_token, match], axis=1)
                self.c_mask = tf.concat([tf.ones(shape=(bs, 1), dtype=tf.bool), self.c_mask], axis=1)
            logits1, logits2 = pointer(init, match, self.hidden_size, self.c_mask)

        with tf.variable_scope("predict"):
            max_ans_length = tf.cast(tf.minimum(15, self.c_maxlen), tf.int64)
            outer_logits = tf.exp(tf.expand_dims(logits1, axis=2) + tf.expand_dims(logits2, axis=1))
            outer_logits = tf.matrix_band_part(outer_logits, 0, max_ans_length)
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, max_ans_length)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            self.yp_logits = tf.reduce_max(tf.reduce_max(outer_logits, axis=2), axis=1)
            if self.noans_token:
                self.yp_score = 1 - tf.nn.softmax(logits1)[:, 0] * tf.nn.softmax(logits2)[:, 0]
            loss_1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=self.y1)
            loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(loss_1 + loss_2)

    def _init_placeholders(self):
        self.c_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='c_ph')
        self.cc_ph = tf.placeholder(shape=(None, None, self.char_limit), dtype=tf.int32, name='cc_ph')
        self.q_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='q_ph')
        self.qc_ph = tf.placeholder(shape=(None, None, self.char_limit), dtype=tf.int32, name='qc_ph')
        self.y1_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y1_ph')
        self.y2_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='y2_ph')

        self.lear_rate_ph = tf.placeholder_with_default(0.0, shape=[], name='learning_rate')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _init_optimizer(self):
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.train_op = self.get_train_op(self.loss, learning_rate=self.lear_rate_ph)

    def _build_feed_dict(self, c_tokens, c_chars, q_tokens, q_chars, y1=None, y2=None):
        feed_dict = {
            self.c_ph: c_tokens,
            self.cc_ph: c_chars,
            self.q_ph: q_tokens,
            self.qc_ph: q_chars,
        }
        if y1 is not None and y2 is not None:
            feed_dict.update({
                self.y1_ph: y1,
                self.y2_ph: y2,
                self.lear_rate_ph: max(self.get_learning_rate(), self.min_learning_rate),
                self.keep_prob_ph: self.keep_prob,
                self.is_train_ph: True,
            })

        return feed_dict

    def train_on_batch(self, c_tokens: np.ndarray, c_chars: np.ndarray, q_tokens: np.ndarray, q_chars: np.ndarray,
                       y1s: Tuple[List[int], ...], y2s: Tuple[List[int], ...]) -> float:
        """
        This method is called by trainer to make one training step on one batch.

        Args:
            c_tokens: batch of tokenized contexts
            c_chars: batch of tokenized contexts, each token split on chars
            q_tokens: batch of tokenized questions
            q_chars: batch of tokenized questions, each token split on chars
            y1s: batch of ground truth answer start positions
            y2s: batch of ground truth answer end positions

        Returns:
            value of loss function on batch
        """
        # TODO: filter examples in batches with answer position greater self.context_limit
        # select one answer from list of correct answers
        y1s = np.array([x[0] for x in y1s])
        y2s = np.array([x[0] for x in y2s])
        if self.noans_token:
            noans_mask = ((y1s != -1) * (y2s != -1))
            y1s = (y1s + 1) * noans_mask
            y2s = (y2s + 1) * noans_mask

        feed_dict = self._build_feed_dict(c_tokens, c_chars, q_tokens, q_chars, y1s, y2s)
        loss, _, lear_rate = self.sess.run([self.loss, self.train_op, self.lear_rate_ph],
                                           feed_dict=feed_dict)
        report = {'loss': loss, 'learning_rate': float(lear_rate), 'momentum': self.get_momentum()}
        return report

    def __call__(self, c_tokens: np.ndarray, c_chars: np.ndarray, q_tokens: np.ndarray, q_chars: np.ndarray,
                 *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Predicts answer start and end positions by given context and question.

        Args:
            c_tokens: batch of tokenized contexts
            c_chars: batch of tokenized contexts, each token split on chars
            q_tokens: batch of tokenized questions
            q_chars: batch of tokenized questions, each token split on chars

        Returns:
            answer_start, answer_end positions, answer logits which represent models confidence
        """
        if any(np.sum(c_tokens, axis=-1) == 0) or any(np.sum(q_tokens, axis=-1) == 0):
            logger.info('SQuAD model: Warning! Empty question or context was found.')
            noanswers = -np.ones(shape=(c_tokens.shape[0]), dtype=np.int32)
            zero_probs = np.zeros(shape=(c_tokens.shape[0]), dtype=np.float32)
            if self.noans_token:
                return noanswers, noanswers, zero_probs, zero_probs
            return noanswers, noanswers, zero_probs

        feed_dict = self._build_feed_dict(c_tokens, c_chars, q_tokens, q_chars)

        if self.noans_token:
            yp1, yp2, logits, score = self.sess.run([self.yp1, self.yp2, self.yp_logits, self.yp_score],
                                                    feed_dict=feed_dict)
            noans_mask = (yp1 * yp2).astype(bool)
            yp1 = yp1 * noans_mask - 1
            yp2 = yp2 * noans_mask - 1
            return yp1, yp2, logits.tolist(), score.tolist()

        yp1, yp2, logits = self.sess.run([self.yp1, self.yp2, self.yp_logits], feed_dict=feed_dict)
        return yp1, yp2, logits.tolist()

    def process_event(self, event_name: str, data) -> None:
        """
        Processes events sent by trainer. Implements learning rate decay.

        Args:
            event_name: event_name sent by trainer
            data: number of examples, epochs, metrics sent by trainer
        """
        super().process_event(event_name, data)
