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

import copy
from typing import Dict
import collections
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

from deeppavlov.core.layers import tf_attention_mechanisms as am
from deeppavlov.core.layers import tf_layers
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('go_bot_rnn')
class GoalOrientedBotNetwork(TFModel):
    """
    The ``GoalOrientedBotNetwork`` is a recurrent network that handles dialogue policy
    management.
    Inputs features of an utterance and predicts label of a bot action
    (classification task).

    An LSTM with a dense layer for input features and a dense layer for it's output.
    Softmax is used as an output activation function.

    Parameters:
        hidden_size: size of rnn hidden layer.
        action_size: size of rnn output (equals to number of bot actions).
        obs_size: input features' size (must be equal to sum of output sizes of
            ``bow_embedder``, ``embedder``, ``intent_classifier``,
            ``tracker.num_features`` plus size of context features(=6) and
            ``action_size``).
        learning_rate: learning rate during training.
        end_learning_rate: if set, learning rate starts from ``learning rate`` value and
            decays polynomially to the value of ``end_learning_rate``.
        decay_steps: number of steps for learning rate to decay.
        decay_power: power used to calculate learning rate decay for polynomial strategy.
        dropout_rate: probability of weights dropping out.
        l2_reg_coef: l2 regularization weight (applied to input and output layer).
        dense_size: rnn input size.
        optimizer: one of tf.train.Optimizer subclasses as a string.
        attention_mechanism: describes attention applied to embeddings of input tokens.

            * **type** – type of attention mechanism, possible values are ``'general'``, ``'bahdanau'``, ``'light_general'``, ``'light_bahdanau'``, ``'cs_general'`` and ``'cs_bahdanau'``.
            * **hidden_size** – attention hidden state size.
            * **max_num_tokens** – maximum number of input tokens.
            * **depth** – number of averages used in constrained attentions
              (``'cs_bahdanau'`` or ``'cs_general'``).
            * **action_as_key** – whether to use action from previous timestep as key
              to attention.
            * **intent_as_key** – use utterance intents as attention key or not.
            * **projected_align** – whether to use output projection.
    """
    GRAPH_PARAMS = ["hidden_size", "action_size", "dense_size", "obs_size",
                    "attention_mechanism"]

    def __init__(self,
                 hidden_size: int,
                 action_size: int,
                 obs_size: int,
                 learning_rate: float,
                 end_learning_rate: float = None,
                 decay_steps: int = 1000,
                 decay_power: float = 1.,
                 dropout_rate: float = 0.,
                 l2_reg_coef: float = 0.,
                 dense_size: int = None,
                 optimizer: str = 'AdamOptimizer',
                 attention_mechanism: Dict = None,
                 **kwargs):
        end_learning_rate = end_learning_rate or learning_rate
        dense_size = dense_size or hidden_size

        # specify model options
        self.opt = {
            'hidden_size': hidden_size,
            'action_size': action_size,
            'obs_size': obs_size,
            'dense_size': dense_size,
            'learning_rate': learning_rate,
            'end_learning_rate': end_learning_rate,
            'decay_steps': decay_steps,
            'decay_power': decay_power,
            'dropout_rate': dropout_rate,
            'l2_reg_coef': l2_reg_coef,
            'optimizer': optimizer,
            'attention_mechanism': attention_mechanism
        }

        # initialize parameters
        self._init_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        super().__init__(**kwargs)
        if tf.train.checkpoint_exists(str(self.save_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

        self.reset_state()

    def __call__(self, features, emb_context, key, action_mask, prob=False):
        feed_dict = {
            self._features: features,
            self._dropout_keep_prob: 1.,
            self._learning_rate: 1.,
            self._utterance_mask: [[1.]],
            self._initial_state: (self.state_c, self.state_h),
            self._action_mask: action_mask
        }
        if self.attn:
            feed_dict[self._emb_context] = emb_context
            feed_dict[self._key] = key

        probs, prediction, state =\
            self.sess.run([self._probs, self._prediction, self._state],
                          feed_dict=feed_dict)

        self.state_c, self._state_h = state
        if prob:
            return probs
        return prediction

    def train_on_batch(self, features, emb_context, key, utter_mask, action_mask, action):
        feed_dict = {
            self._dropout_keep_prob: 1 - self.dropout_rate,
            self._learning_rate: self.get_learning_rate(), 
            self._utterance_mask: utter_mask,
            self._features: features,
            self._action: action,
            self._action_mask: action_mask
        }
        if self.attn:
            feed_dict[self._emb_context] = emb_context
            feed_dict[self._key] = key

        _, loss_value, prediction = \
            self.sess.run([self._train_op, self._loss, self._prediction],
                          feed_dict=feed_dict)
        return loss_value, prediction

    def _init_params(self):
        self.learning_rate = self.opt['learning_rate']
        self.end_learning_rate = self.opt['end_learning_rate']
        self.decay_steps = self.opt['decay_steps']
        self.decay_power = self.opt['decay_power']
        self.dropout_rate = self.opt['dropout_rate']
        self.hidden_size = self.opt['hidden_size']
        self.action_size = self.opt['action_size']
        self.obs_size = self.opt['obs_size']
        self.dense_size = self.opt['dense_size']
        self.l2_reg = self.opt['l2_reg_coef']

        self._optimizer = None
        if hasattr(tf.train, self.opt['optimizer']):
            self._optimizer = getattr(tf.train, self.opt['optimizer'])
        if not issubclass(self._optimizer, tf.train.Optimizer):
            raise ConfigError("`optimizer` parameter should be a name of"
                              " tf.train.Optimizer subclass")

        attn = self.opt.get('attention_mechanism')
        if attn:
            self.opt['attention_mechanism'] = attn

            self.attn = \
                collections.namedtuple('attention_mechanism', attn.keys())(**attn)
            self.obs_size -= attn['token_size']
        else:
            self.attn = None

    def _build_graph(self):

        self._add_placeholders()

        # build body
        _logits, self._state = self._build_body()

        # probabilities normalization : elemwise multiply with action mask
        _logits_exp = tf.multiply(tf.exp(_logits), self._action_mask)
        _logits_exp_sum = tf.expand_dims(tf.reduce_sum(_logits_exp, -1), -1)
        self._probs = tf.squeeze(_logits_exp / _logits_exp_sum, name='probs')

        # loss, train and predict operations
        self._prediction = tf.argmax(self._probs, axis=-1, name='prediction')

        _weights = tf.expand_dims(self._utterance_mask, -1)
        # TODO: try multiplying logits to action_mask
        # onehots = tf.one_hot(self._action, self.action_size)
        # _loss_tensor = \
        # tf.losses.softmax_cross_entropy(logits=_logits, onehot_labels=onehots,
        #                                weights=_weights,
        #                                reduction=tf.losses.Reduction.NONE)
        _loss_tensor = tf.losses.sparse_softmax_cross_entropy(
            logits=_logits, labels=self._action, weights=_weights,
            reduction=tf.losses.Reduction.NONE
        )
        # multiply with batch utterance mask
        # _loss_tensor = tf.multiply(_loss_tensor, self._utterance_mask)
        self._loss = tf.reduce_mean(_loss_tensor, name='loss')
        self._loss += self.l2_reg * tf.losses.get_regularization_loss()
        self._train_op = self.get_train_op(self._loss,
                                           learning_rate=self._learning_rate,
                                           optimizer=self._optimizer,
                                           clip_norm=2.)

    def _add_placeholders(self):
        self._dropout_keep_prob = tf.placeholder_with_default(1.0,
                                                              shape=[],
                                                              name='dropout_prob')
        self._learning_rate = tf.placeholder(tf.float32,
                                             shape=[],
                                             name='learning_rate')
        self._features = tf.placeholder(tf.float32,
                                        [None, None, self.obs_size],
                                        name='features')
        self._action = tf.placeholder(tf.int32,
                                      [None, None],
                                      name='ground_truth_action')
        self._action_mask = tf.placeholder(tf.float32,
                                           [None, None, self.action_size],
                                           name='action_mask')
        self._utterance_mask = tf.placeholder(tf.float32,
                                              shape=[None, None],
                                              name='utterance_mask')
        _batch_size = tf.shape(self._features)[0]
        zero_state = tf.zeros([_batch_size, self.hidden_size], dtype=tf.float32)
        _initial_state_c = \
            tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])
        _initial_state_h = \
            tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c,
                                                            _initial_state_h)
        if self.attn:
            _emb_context_shape = \
                [None, None, self.attn.max_num_tokens, self.attn.token_size]
            self._emb_context = tf.placeholder(tf.float32,
                                               _emb_context_shape,
                                               name='emb_context')
            self._key = tf.placeholder(tf.float32, 
                                       [None, None, self.attn.key_size],
                                       name='key')

    def _build_body(self):
        # input projection
        _units = tf.layers.dense(self._features, self.dense_size,
                                 kernel_regularizer=tf.nn.l2_loss,
                                 kernel_initializer=xav())
        if self.attn:
            attn_scope = "attention_mechanism/{}".format(self.attn.type)
            with tf.variable_scope(attn_scope):
                if self.attn.type == 'general':
                    _attn_output = am.general_attention(
                        self._key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        projected_align=self.attn.projected_align)
                elif self.attn.type == 'bahdanau':
                    _attn_output = am.bahdanau_attention(
                        self._key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        projected_align=self.attn.projected_align)
                elif self.attn.type == 'cs_general':
                    _attn_output = am.cs_general_attention(
                        self._key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        depth=self.attn.depth,
                        projected_align=self.attn.projected_align)
                elif self.attn.type == 'cs_bahdanau':
                    _attn_output = am.cs_bahdanau_attention(
                        self._key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        depth=self.attn.depth,
                        projected_align=self.attn.projected_align)
                elif self.attn.type == 'light_general':
                    _attn_output = am.light_general_attention(
                        self._key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        projected_align=self.attn.projected_align)
                elif self.attn.type == 'light_bahdanau':
                    _attn_output = am.light_bahdanau_attention(
                        self._key,
                        self._emb_context,
                        hidden_size=self.attn.hidden_size,
                        projected_align=self.attn.projected_align)
                else:
                    raise ValueError("wrong value for attention mechanism type")
            _units = tf.concat([_units, _attn_output], -1)

        _units = tf_layers.variational_dropout(_units,
                                               keep_prob=self._dropout_keep_prob)

        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        _utter_lengths = tf.to_int32(tf.reduce_sum(self._utterance_mask, axis=-1))
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _units,
                                            initial_state=self._initial_state,
                                            sequence_length=_utter_lengths)

        # output projection
        _logits = tf.layers.dense(_output, self.action_size,
                                  kernel_regularizer=tf.nn.l2_loss,
                                  kernel_initializer=xav(), name='logits')
        return _logits, _state

    def get_learning_rate(self):
        # polynomial decay
        global_step = min(self.global_step, self.decay_steps)
        decayed_learning_rate = \
            (self.learning_rate - self.end_learning_rate) *\
            (1 - global_step / self.decay_steps) ** self.decay_power +\
            self.end_learning_rate
        return decayed_learning_rate

    def load(self, *args, **kwargs):
        self.load_params()
        super().load(*args, **kwargs)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        path = str(self.save_path.with_suffix('.json').resolve())
        log.info('[saving parameters to {}]'.format(path))
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(self.opt, fp)

    def load_params(self):
        path = str(self.load_path.with_suffix('.json').resolve())
        log.info('[loading parameters from {}]'.format(path))
        with open(path, 'r', encoding='utf8') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt.get(p) != params.get(p):
                raise ConfigError("`{}` parameter must be equal to saved model "
                                  "parameter value `{}`, but is equal to `{}`"
                                  .format(p, params.get(p), self.opt.get(p)))

    def process_event(self, event_name, data):
        if event_name == "after_epoch":
            log.info("Updating global step, learning rate = {:.6f}."
                     .format(self.get_learning_rate()))
            self.global_step += 1

    def reset_state(self):
        # set zero state
        self.state_c = np.zeros([1, self.hidden_size], dtype=np.float32)
        self.state_h = np.zeros([1, self.hidden_size], dtype=np.float32)
        # setting global step number to 0
        self.global_step = 0

    def shutdown(self):
        self.sess.close()
