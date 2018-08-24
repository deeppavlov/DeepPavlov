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

from typing import List, Tuple

import tensorflow as tf
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.models.squad.utils import dot_attention, simple_attention, PtrNet, CudnnGRU, CudnnCompatibleGRU
from deeppavlov.core.common.check_gpu import GPU_AVAILABLE
from deeppavlov.core.layers.tf_layers import cudnn_bi_gru, variational_dropout
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


@register('squad_model')
class SquadModel(TFModel):
    """
    SquadModel predicts answer start and end position in given context by given question.

    High level architecture:
    Word embeddings -> Contextual embeddings -> Question-Context Attention -> Self-attention -> Pointer Network

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
        learning_rate: initial learning rate
        min_learning_rate: min learning rate, is used in learning rate decay
        learning_rate_patience: number of epochs without score improvements to decay learning rate
        grad_clip: gradient clipping value
        weight_decay: weight decay value
    """
    def __init__(self, word_emb: np.ndarray, char_emb: np.ndarray, context_limit: int = 450, question_limit: int = 150,
                 char_limit: int = 16, train_char_emb: bool = True, char_hidden_size: int = 100,
                 encoder_hidden_size: int = 75, attention_hidden_size: int = 75, keep_prob: float = 0.7,
                 learning_rate: float = 0.5, min_learning_rate: float = 0.001, learning_rate_patience: int = 1,
                 grad_clip: float = 5.0, weight_decay: float = 1.0, **kwargs):

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
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay

        self.word_emb_dim = self.init_word_emb.shape[1]
        self.char_emb_dim = self.init_char_emb.shape[1]

        self.last_impatience = 0
        self.lr_impatience = 0

        if GPU_AVAILABLE:
            self.GRU = CudnnGRU
        else:
            self.GRU = CudnnCompatibleGRU

        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.sess_config)

        self._init_graph()

        self._init_optimizer()

        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)
        # Try to load the model (if there are some model files the model will be loaded from them)
        if self.load_path is not None:
            self.load()
            if self.weight_decay < 1.0:
                 self.sess.run(self.assign_vars)

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
        self.y1 = tf.one_hot(self.y1_ph, depth=self.context_limit)
        self.y2 = tf.one_hot(self.y2_ph, depth=self.context_limit)
        self.y1 = tf.slice(self.y1, [0, 0], [bs, self.c_maxlen])
        self.y2 = tf.slice(self.y2, [0, 0], [bs, self.c_maxlen])

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
            logits1, logits2 = pointer(init, match, self.hidden_size, self.c_mask)

        with tf.variable_scope("predict"):
            outer_logits = tf.exp(tf.expand_dims(logits1, axis=2) + tf.expand_dims(logits2, axis=1))
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, tf.cast(tf.minimum(15, self.c_maxlen), tf.int64))
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            self.yp_logits = tf.reduce_max(tf.reduce_max(outer_logits, axis=2), axis=1)
            loss_1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=self.y1)
            loss_2 = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(loss_1 + loss_2)

        if self.weight_decay < 1.0:
            self.var_ema = tf.train.ExponentialMovingAverage(self.weight_decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.shadow_vars = []
                self.global_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.shadow_vars.append(v)
                        self.global_vars.append(var)
                self.assign_vars = []
                for g, v in zip(self.global_vars, self.shadow_vars):
                    self.assign_vars.append(tf.assign(g, v))

    def _init_placeholders(self):
        self.c_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='c_ph')
        self.cc_ph = tf.placeholder(shape=(None, None, self.char_limit), dtype=tf.int32, name='cc_ph')
        self.q_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='q_ph')
        self.qc_ph = tf.placeholder(shape=(None, None, self.char_limit), dtype=tf.int32, name='qc_ph')
        self.y1_ph = tf.placeholder(shape=(None, ), dtype=tf.int32, name='y1_ph')
        self.y2_ph = tf.placeholder(shape=(None, ), dtype=tf.int32, name='y2_ph')

        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name='lr_ph')
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=[], name='keep_prob_ph')
        self.is_train_ph = tf.placeholder_with_default(False, shape=[], name='is_train_ph')

    def _init_optimizer(self):
        with tf.variable_scope('Optimizer'):
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr_ph, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads = [tf.clip_by_norm(g, self.grad_clip) for g in gradients]
            self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

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
                self.lr_ph: self.learning_rate,
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
        y1s = list(map(lambda x: x[0], y1s))
        y2s = list(map(lambda x: x[0], y2s))
        feed_dict = self._build_feed_dict(c_tokens, c_chars, q_tokens, q_chars, y1s, y2s)
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return loss

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
            return noanswers, noanswers

        feed_dict = self._build_feed_dict(c_tokens, c_chars, q_tokens, q_chars)
        yp1, yp2, logits = self.sess.run([self.yp1, self.yp2, self.yp_logits], feed_dict=feed_dict)
        return yp1, yp2, [float(logit) for logit in logits]

    def process_event(self, event_name: str, data) -> None:
        """
        Processes events sent by trainer. Implements learning rate decay.

        Args:
            event_name: event_name sent by trainer
            data: number of examples, epochs, metrics sent by trainer
        """
        if event_name == "after_validation":
            if data['impatience'] > self.last_impatience:
                self.lr_impatience += 1
            else:
                self.lr_impatience = 0

            self.last_impatience = data['impatience']

            if self.lr_impatience >= self.learning_rate_patience:
                self.lr_impatience = 0
                self.learning_rate = max(self.learning_rate / 2, self.min_learning_rate)
                logger.info('SQuAD model: learning_rate changed to {}'.format(self.learning_rate))
            logger.info('SQuAD model: lr_impatience: {}, learning_rate: {}'.format(self.lr_impatience, self.learning_rate))

    def shutdown(self):
        pass
