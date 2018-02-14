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

        save_path = self.opt.get('save_path')
        load_path = self.opt.get('load_path', None)
        train_now = self.opt.get('train_now', False)

        super().__init__(save_path=save_path,
                         load_path=load_path,
                         train_now=train_now,
                         mode=self.opt['mode'])

        # initialize parameters
        self._init_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        if self.get_checkpoint_state():
        #TODO: save/load params to json, here check compatability
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))
            self.sess.run(tf.global_variables_initializer())

        self.reset_state()

    def run_sess(self):
        pass

    def _init_params(self, params=None):
        params = params or self.opt
        self.learning_rate = params['learning_rate']
        self.n_hidden = params['hidden_dim']
        self.n_actions = params['action_size']
        #TODO: try obs_size=None or as a placeholder
        self.obs_size = params['obs_size']
        self.dense_size = params.get('dense_size', params['hidden_dim'])

    def _build_graph(self):

        self._add_placeholders()

        # build body
        _logits, self._state = self._build_body()

        # probabilities normalization : elemwise multiply with action mask
        self._probs = tf.squeeze(tf.nn.softmax(_logits))
        #TODO: add action mask
        #self._probs = tf.multiply(tf.squeeze(tf.nn.softmax(_logits)),
        #                          self._action_mask,
        #                          name='probs')

        # loss, train and predict operations
        self._prediction = tf.argmax(self._probs, axis=-1, name='prediction')
        _loss_tensor = \
            tf.losses.sparse_softmax_cross_entropy(logits=_logits,
                                                   labels=self._action)
        self._loss = tf.reduce_mean(_loss_tensor, name='loss')
        self._train_op = self._get_train_op(self._loss, self.learning_rate)

    def _add_placeholders(self):
        # TODO: make batch_size != 1
        _initial_state_c = \
            tf.placeholder_with_default(np.zeros([1, self.n_hidden], np.float32),
                                        shape=[1, self.n_hidden])
        _initial_state_h = \
            tf.placeholder_with_default(np.zeros([1, self.n_hidden], np.float32),
                                        shape=[1, self.n_hidden])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c,
                                                            _initial_state_h)
        self._features = tf.placeholder(tf.float32, [1, None, self.obs_size],
                                        name='features')
        self._action = tf.placeholder(tf.int32, [1, None],
                                      name='ground_truth_action')
        self._action_mask = tf.placeholder(tf.float32, [1, None, self.n_actions],
                                           name='action_mask')

    def _build_body(self):
        # input projection
        _projected_features = \
            tf.layers.dense(self._features,
                            self.dense_size,
                            kernel_initializer=xavier_initializer())

        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _projected_features,
                                            initial_state=self._initial_state)
 
        # output projection
        # TODO: try multiplying logits to action_mask
        _logits = tf.layers.dense(_output,
                                  self.n_actions,
                                  kernel_initializer=xavier_initializer())
        return _logits, _state

    def _get_train_op(self, loss, learning_rate, optimizer=None, clip_norm=1.):
        """ Get train operation for given loss

        Args:
            loss: loss function, tf tensor or scalar
            learning_rate: scalar or placeholder
            optimizer: instance of tf.train.Optimizer, Adam, by default

        Returns:
            train_op
        """

        optimizer = optimizer or tf.train.AdamOptimizer
        optimizer = optimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
        grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var)\
                          for grad, var in grads_and_vars]
        return optimizer.apply_gradients(grads_and_vars, name='train_op')

    def reset_state(self):
        # set zero state
        self.state_c = np.zeros([1, self.n_hidden], dtype=np.float32)
        self.state_h = np.zeros([1, self.n_hidden], dtype=np.float32)

    def _train_step(self, features, action, action_mask):
        _, loss_value, prediction = \
            self.sess.run(
                [ self._train_op, self._loss, self._prediction ],
                feed_dict={
                    self._features: [features],
                    self._action: [action],
                    self._action_mask: [action_mask]
                }
            )
        return loss_value, prediction

    def _forward(self, features, action_mask, prob=False):
        probs, prediction, state = \
            self.sess.run(
                [ self._probs, self._prediction, self._state ],
                feed_dict={
                    self._features: [[features]],
                    self._initial_state: (self.state_c, self.state_h),
                    self._action_mask: [[action_mask]]
                }
            )
        self.state_c, self._state_h = state
        if prob:
            return probs
        return prediction

    def shutdown(self):
        self.sess.close()
