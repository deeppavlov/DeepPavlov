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


from deeppavlov.skills.go_bot import csoftmax_attention
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

    def __call__(self, features, emb_context, key, action_mask, prob=False):
        # TODO: make input list
        probs, prediction, state = \
            self.sess.run(
                [self._probs, self._prediction, self._state],
                feed_dict={
                    self._features: [[features]],
                    self._emb_context: [[emb_context]],
                    self._key: [[key]],
                    self._initial_state: (self.state_c, self.state_h),
                    self._action_mask: [[action_mask]]
                }
            )
        self.state_c, self._state_h = state
        if prob:
            return probs
        return prediction

    def train_on_batch(self, x: list, y: list):
        features, emb_context, key, action_mask = x
        action = y
        self._train_step(features, emb_context, key, action, action_mask)

    def _init_params(self, params=None):
        params = params or self.opt
        self.learning_rate = params['learning_rate']
        self.n_hidden = params['hidden_dim']
        self.n_att_hidden = params['att_hidden_dim']
        self.n_actions = params['action_size']
        #TODO: try obs_size=None or as a placeholder
        self.obs_size = params['obs_size']
        self.max_of_context_tokens = params['max_of_context_tokens']
        self.token_dim = params['token_dim']
        self.key_dim = params['key_dim']
        self.attention_mechanism = params['attention_mechanism']
        self.attention_depth = params.get('attention_depth')
        self.batch_size = 1
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
        self._train_op = self.get_train_op(self._loss, self.learning_rate)

    def _add_placeholders(self):
        # TODO: make batch_size != 1
        _initial_state_c = \
            tf.placeholder_with_default(np.zeros([self.batch_size, self.n_hidden], np.float32),
                                        shape=[self.batch_size, self.n_hidden])
        _initial_state_h = \
            tf.placeholder_with_default(np.zeros([self.batch_size, self.n_hidden], np.float32),
                                        shape=[self.batch_size, self.n_hidden])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c,
                                                            _initial_state_h)
        self._features = tf.placeholder(tf.float32, [self.batch_size, None, self.obs_size],
                                        name='features')
        self._emb_context = tf.placeholder(tf.float32, [self.batch_size, None, self.max_of_context_tokens, self.token_dim],
                                        name='emb_context')
        self._key = tf.placeholder(tf.float32, [self.batch_size, None, self.key_dim],
                                        name='key')
        self._action = tf.placeholder(tf.int32, [self.batch_size, None],
                                      name='ground_truth_action')
        self._action_mask = tf.placeholder(tf.float32, [self.batch_size, None, self.n_actions],
                                           name='action_mask')

    def _build_body(self):
        # input projection
        _projected_features = \
            tf.layers.dense(self._features,
                            self.dense_size,
                            kernel_initializer=xavier_initializer())

        if self.attention_mechanism == 'general':
            _att_mech_output_tensor = self._general_att_mech()
        elif self.attention_mechanism == 'cs_general':
            _att_mech_output_tensor = self._cs_general_att_mech()

        _concatenated_features = tf.concat([_projected_features, _att_mech_output_tensor],-1)
        # _concatenated_features = _projected_features


        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _concatenated_features,
                                            initial_state=self._initial_state)

        # output projection
        # TODO: try multiplying logits to action_mask
        _logits = tf.layers.dense(_output,
                                  self.n_actions,
                                  kernel_initializer=xavier_initializer())
        return _logits, _state

    def _general_att_mech(self):
        with tf.name_scope("attention_mechanism/general"):

            _raw_key = self._key
            _n_hidden = (self.n_att_hidden//2)*2
            _batch_size = self.batch_size
            _max_of_context_tokens = self.max_of_context_tokens
            _token_dim = self.token_dim
            _context = self._emb_context

            _context_dim = tf.shape(_context)
            _r_context = tf.reshape(_context, shape = [-1, _max_of_context_tokens, _token_dim])

            _projected_key = \
                    tf.layers.dense(_raw_key,
                                _n_hidden,
                                kernel_initializer=xavier_initializer()) # [self.batch_size, None, _n_hidden]
            _projected_key_dim = tf.shape(_projected_key)
            _r_projected_key = tf.reshape(_projected_key, shape = [-1, _n_hidden, 1])

            _lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            _lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            (_output_fw, _output_bw), _states = \
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=_lstm_fw_cell,
                                                    cell_bw=_lstm_bw_cell,
                                                    inputs=_r_context,
                                                    dtype=tf.float32)
            _bilstm_output = tf.concat([_output_fw, _output_bw],-1) # [-1,self.max_of_context_tokens,_n_hidden])

            _score = tf.nn.softmax(tf.matmul(_bilstm_output,_r_projected_key))

            _t_bilstm_output = tf.transpose(_bilstm_output, [0, 2, 1])
            _output_tensor = tf.reshape(tf.matmul(_t_bilstm_output,_score), shape = [_batch_size, -1, _n_hidden])
        return _output_tensor

    def _cs_general_att_mech(self):
        with tf.name_scope("attention_mechanism/cs_general"):

            _raw_key = self._key
            _attention_depth = self.attention_depth
            _n_hidden = (self.n_att_hidden//2)*2
            _batch_size = self.batch_size
            _max_of_context_tokens = self.max_of_context_tokens
            _token_dim = self.token_dim
            _key_dim = self.key_dim
            _context = self._emb_context

            _context_dim = tf.shape(_context)
            _r_context = tf.reshape(_context, shape = [-1, _max_of_context_tokens, _token_dim])
            assert _attention_depth is not None


            _lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            _lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            (_output_fw, _output_bw), _states = \
                    tf.nn.bidirectional_dynamic_rnn(cell_fw=_lstm_fw_cell,
                                                    cell_bw=_lstm_bw_cell,
                                                    inputs=_r_context,
                                                    dtype=tf.float32)
            _bilstm_output = tf.concat([_output_fw, _output_bw],-1) # [-1,self.max_of_context_tokens,_n_hidden])

            _key = tf.reshape(_raw_key, [-1, _key_dim])
            _final_sketch = csoftmax_attention.attention_block(_bilstm_output, _raw_key, _attention_depth)

            _output_tensor = tf.reshape(_final_sketch, shape = [_batch_size, -1, _attention_depth * _n_hidden])
        return _output_tensor

    def reset_state(self):
        # set zero state
        self.state_c = np.zeros([1, self.n_hidden], dtype=np.float32)
        self.state_h = np.zeros([1, self.n_hidden], dtype=np.float32)

    def _train_step(self, features, emb_context, key, action, action_mask):
        _, loss_value, prediction = \
            self.sess.run(
                [ self._train_op, self._loss, self._prediction ],
                feed_dict={
                    self._features: [features],
                    self._emb_context: [emb_context],
                    self._key: [key],
                    self._action:[action],
                    self._action_mask: [action_mask]
                }
            )
        return loss_value, prediction

    def shutdown(self):
        self.sess.close()
