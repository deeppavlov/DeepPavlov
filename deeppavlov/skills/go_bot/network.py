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
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('go_bot_rnn')
class GoalOrientedBotNetwork(TFModel):
    def __init__(self, **params):
        self.opt = params

        # initialize parameters
        self._init_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        super().__init__(**params)
        if tf.train.checkpoint_exists(str(self.save_path.resolve())):
        #TODO: save/load params to json, here check compatability
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

        self.reset_state()

    def __call__(self, features, action_mask, prob=False):
        # TODO: make input list
        # TODO: batch_size != 1
        #batch_size = len(features)
        probs, prediction, state = \
            self.sess.run(
                [self._probs, self._prediction, self._state],
                feed_dict={
                    self._dropout: 1.,
                    self._utterance_mask: [[1.]],
                    self._features: features,
                    self._initial_state: (self.state_c, self.state_h),
                    self._action_mask: action_mask
                }
            )
        self.state_c, self._state_h = state
        if prob:
            return probs
        return prediction

    def train_on_batch(self, x: list, y: list):
        features, utter_mask, action_mask = x
        action = y
        self._train_step(features, utter_mask, action_mask, action)

    def _init_params(self, params=None):
        params = params or self.opt
        self.learning_rate = params['learning_rate']
        self.dropout_rate = params.get('dropout_rate', 1.)
        self.n_hidden = params['hidden_dim']
        self.n_actions = params['action_size']
        self.obs_size = params['obs_size']
        self.dense_size = params.get('dense_size', params['hidden_dim'])

    def _build_graph(self):

        self._add_placeholders()

        # build body
        _logits, self._state = self._build_body()
        # multiply with batch utteranct mask
        #_logits = tf.multiply(_logits, tf.expand_dims(self._utterance_mask, -1))
        # TODO: try multiplying logits to action_mask

        # probabilities normalization : elemwise multiply with action mask
        #self._probs = tf.multiply(_logits, self._action_mask)
        #self._probs = tf.squeeze(tf.nn.softmax(self._probs), name='probs')
        self._probs = tf.squeeze(tf.nn.softmax(_logits), name='probs')

        # loss, train and predict operations
        self._prediction = tf.argmax(self._probs, axis=-1, name='prediction')
        #_weights=tf.expand_dims((self._utterance_mask + 1) % 2, -1)
        _weights = tf.expand_dims(self._utterance_mask, -1)
        _loss_tensor = \
            tf.losses.sparse_softmax_cross_entropy(logits=_logits,
                                                   labels=self._action,
                                                   reduction="none")
        print(_loss_tensor)
        _loss_tensor = tf.multiply(_loss_tensor, self._utterance_mask)
        self._loss = tf.reduce_mean(_loss_tensor, name='loss')
        self._train_op = self.get_train_op(self._loss, self.learning_rate, clip_norm=2.)

    def _add_placeholders(self):
        # TODO: make batch_size != 1
        self._dropout = tf.placeholder_with_default(1.0, shape=[])
        self._features = tf.placeholder(tf.float32,
                                        [None, None, self.obs_size],
                                        name='features')
        self._action = tf.placeholder(tf.int32,
                                      [None, None],
                                      name='ground_truth_action')
        self._action_mask = tf.placeholder(tf.float32,
                                           [None, None, self.n_actions],
                                           name='action_mask')
        #self._utterance_mask = tf.placeholder_with_default([[1.]],
        self._utterance_mask = tf.placeholder(tf.float32,
                                              shape=[None, None],
                                              name='utterance_mask')
        _initial_state_c = \
            tf.placeholder_with_default(np.zeros([1, self.n_hidden], np.float32),
                                        shape=[None, self.n_hidden])
        _initial_state_h = \
            tf.placeholder_with_default(np.zeros([1, self.n_hidden], np.float32),
                                        shape=[None, self.n_hidden])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c,
                                                            _initial_state_h)
        #batch_size = tf.shape(self._features)[0]
        #self._initial_state = \
            #tf.concat([tf.tile(_initial_state_c, [batch_size, 1]),
            #           tf.tile(_initial_state_h, [batch_size, 1])],
            #          axis=0)

    def _build_body(self):
        # input projection
        _units = tf.nn.dropout(self._features, self._dropout)
        _units = tf.layers.dense(_units,
                                 self.dense_size,
                                 kernel_initializer=xavier_initializer())

        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _units,
                                            initial_state=self._initial_state)
 
        # output projection
        _logits = tf.layers.dense(_output,
                                  self.n_actions,
                                  kernel_initializer=xavier_initializer())
        return _logits, _state

    def reset_state(self):
        # set zero state
        self.state_c = np.zeros([1, self.n_hidden], dtype=np.float32)
        self.state_h = np.zeros([1, self.n_hidden], dtype=np.float32)

    def _train_step(self, features, utter_mask, action_mask, action):
        batch_size = len(features)
        #print("Train step received batch_size =", batch_size)
        #print(np.asarray(features).shape, np.asarray(utter_mask).shape,
        #      np.asarray(action_mask).shape, np.asarray(action).shape)
        #print("Utter_mask =", np.asarray(utter_mask))
        #print("action mask =", np.asarray(action_mask))
        #print("Action =", np.asarray(action))
        _, loss_value, prediction = \
            self.sess.run(
                [ self._train_op, self._loss, self._prediction ],
                feed_dict={
                    self._dropout: self.dropout_rate,
                    self._utterance_mask: utter_mask,
                    self._initial_state: (np.tile(self.state_c, [batch_size, 1]),
                                          np.tile(self.state_h, [batch_size, 1])),
                    self._features: features,
                    self._action: action,
                    self._action_mask: action_mask
                }
            )
        return loss_value, prediction

    def shutdown(self):
        self.sess.close()
