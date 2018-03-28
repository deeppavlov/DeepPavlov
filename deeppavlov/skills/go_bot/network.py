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
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

import collections

from deeppavlov.skills.go_bot import csoftmax_attention
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('go_bot_rnn')
class GoalOrientedBotNetwork(TFModel):
    GRAPH_PARAMS = ["hidden_size", "action_size", "dense_size", "obs_size"]

    def __init__(self, **params):
        self.debug_pipe = None
        # initialize parameters
        self._init_params(params)
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        super().__init__(**params)
        if tf.train.checkpoint_exists(str(self.save_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

        self.reset_state()

    def __call__(self, features, emb_context, key, action_mask, prob=False):
        feed_dict = {
            self._features: features,
            self._dropout: 1.,
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
            self._dropout: self.dropout_rate,
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


    def _init_params(self, params):
        self.opt = params
        self.opt['dropout_rate'] = params.get('dropout_rate', 1.)
        self.opt['dense_size'] = params.get('dense_size', self.opt['hidden_size'])

        self.learning_rate = self.opt['learning_rate']
        self.dropout_rate = self.opt['dropout_rate']
        self.hidden_size = self.opt['hidden_size']
        self.action_size = self.opt['action_size']
        self.obs_size = self.opt['obs_size']
        self.dense_size = self.opt['dense_size']

        attn = params.get('attention_mechanism')
        if attn:
            attn['intent_dim'] = attn.get('intent_dim', 0)
            attn['key_dim'] = attn['intent_dim'] + attn['key_dim']
            self.attn = \
                collections.namedtuple('attention_mechanism', attn.keys())(**attn)
            self.obs_size = self.attn.obs_size_correction
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
        #onehots = tf.one_hot(self._action, self.action_size)
        #_loss_tensor = \
        # tf.losses.softmax_cross_entropy(logits=_logits, onehot_labels=onehots,
        #                                weights=_weights,
        #                                reduction=tf.losses.Reduction.NONE)
        _loss_tensor = \
            tf.losses.sparse_softmax_cross_entropy(logits=_logits,
                                                   labels=self._action,
                                                   weights=_weights,
                                                   reduction=tf.losses.Reduction.NONE)
        # multiply with batch utterance mask
        #_loss_tensor = tf.multiply(_loss_tensor, self._utterance_mask)
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
                                           [None, None, self.action_size],
                                           name='action_mask')
        self._utterance_mask = tf.placeholder(tf.float32,
                                              shape=[None, None],
                                              name='utterance_mask')
        _batch_size = tf.shape(self._features)[0]
        zero_state = tf.zeros([_batch_size, self.hidden_size], dtype=tf.float32)
        _initial_state_c = tf.placeholder_with_default(zero_state,
                                                       shape=[None, self.hidden_size])
        _initial_state_h = tf.placeholder_with_default(zero_state,
                                                       shape=[None, self.hidden_size])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c,
                                                            _initial_state_h)
        if self.attn:
            _emb_context_shape = \
                [None, None, self.attn.max_of_context_tokens, self.attn.token_dim]
            self._emb_context = tf.placeholder(tf.float32,
                                               _emb_context_shape,
                                               name='emb_context')
            self._key = tf.placeholder(tf.float32, 
                                       [None, None, self.attn.key_dim],
                                       name='key')

    def _build_body(self):
        # input projection
        _units = tf.nn.dropout(self._features, self._dropout)
        _units = tf.layers.dense(_units, self.dense_size,
                                 kernel_initializer=xav(), name='units')

        if self.attn:
            if self.attn.type == 'general':
                _att_mech_output_tensor = self._general_att_mech()
            elif self.attn.type == 'bahdanau':
                _att_mech_output_tensor = self._bahdanau_att_mech()
            elif self.attn.type == 'cs_general':
                _att_mech_output_tensor = self._cs_general_att_mech()
            elif self.attn.type == 'cs_bahdanau':
                _att_mech_output_tensor = self._cs_bah_att_mech()
            elif self.attn.type == 'light_general':
                _att_mech_output_tensor = self._light_general_att_mech()
            elif self.attn.type == 'light_bahdanau':
                _att_mech_output_tensor = self._light_bahdanau_att_mech()
            _units = tf.concat([_units, _att_mech_output_tensor], -1)

        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        _utter_lengths = tf.to_int32(tf.reduce_sum(self._utterance_mask, axis=-1))
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _units,
                                            initial_state=self._initial_state,
                                            sequence_length=_utter_lengths)

        # output projection
        _logits = tf.layers.dense(_output, self.action_size,
                                  kernel_initializer=xav(), name='logits')
        return _logits, _state

    def _general_att_mech(self):
        with tf.variable_scope("attention_mechanism/general"):

            _raw_key = self._key
            _n_hidden = (self.attn.att_hidden_dim//2)*2
            _max_of_context_tokens = self.attn.max_of_context_tokens
            _token_dim = self.attn.token_dim
            _context = self._emb_context
            _projected_attn_alignment = self.attn.projected_attn_alignment

            _context_dim = tf.shape(_context)
            _batch_size = _context_dim[0]
            _r_context = \
                tf.reshape(_context, shape=[-1, _max_of_context_tokens, _token_dim])

            # [None, None, _n_hidden]
            _projected_key = tf.layers.dense(_raw_key, _n_hidden,
                                             kernel_initializer=xav())
            _projected_key_dim = tf.shape(_projected_key)
            _r_projected_key = tf.reshape(_projected_key, shape=[-1, _n_hidden, 1])

            _lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            _lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            (_output_fw, _output_bw), _states = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=_lstm_fw_cell,
                                                cell_bw=_lstm_bw_cell,
                                                inputs=_r_context,
                                                dtype=tf.float32)
            # [-1,self.max_of_context_tokens,_n_hidden])
            _bilstm_output = tf.concat([_output_fw, _output_bw], -1)

            _attn = tf.nn.softmax(tf.matmul(_bilstm_output, _r_projected_key), dim=1)

            if _projected_attn_alignment:
                log.info("Using projected attnention alignment")
                _t_context = tf.transpose(_bilstm_output, [0, 2, 1])
                _output_tensor = tf.reshape(tf.matmul(_t_context, _attn),
                                            shape=[_batch_size, -1, _n_hidden])
            else:
                log.info("Using without projected attnention alignment")
                _t_context = tf.transpose(_r_context, [0, 2, 1])
                _output_tensor = tf.reshape(tf.matmul(_t_context, _attn),
                                            shape=[_batch_size, -1, _token_dim])
        return _output_tensor

    def _light_general_att_mech(self):
        with tf.variable_scope("attention_mechanism/light_general"):

            _raw_key = self._key
            _n_hidden = (self.attn.att_hidden_dim//2)*2
            _max_of_context_tokens = self.attn.max_of_context_tokens
            _token_dim = self.attn.token_dim
            _context = self._emb_context
            _projected_attn_alignment = self.attn.projected_attn_alignment

            _context_dim = tf.shape(_context)
            _batch_size = _context_dim[0]
            _r_context = \
                tf.reshape(_context, shape=[-1, _max_of_context_tokens, _token_dim])

            # [None, None, _n_hidden]
            _projected_key = \
                tf.layers.dense(_raw_key, _n_hidden, kernel_initializer=xav())

            # [None, None, _n_hidden]
            _projected_context = \
                tf.layers.dense(_r_context, _n_hidden, kernel_initializer=xav())
            _projected_key_dim = tf.shape(_projected_key)
            _r_projected_key = tf.reshape(_projected_key, shape=[-1, _n_hidden, 1])

            _attn = tf.nn.softmax(tf.matmul(_projected_context, _r_projected_key), dim=1)

            _t_context = tf.transpose(_r_context, [0, 2, 1])
            _output_tensor = tf.reshape(tf.matmul(_t_context, _attn),
                                        shape=[_batch_size, -1, _token_dim])
        return _output_tensor

    def _light_bahdanau_att_mech(self):
        with tf.variable_scope("attention_mechanism/light_general"):

            _raw_key = self._key
            _n_hidden = (self.attn.att_hidden_dim//2)*2
            _max_of_context_tokens = self.attn.max_of_context_tokens
            _token_dim = self.attn.token_dim
            _context = self._emb_context
            _projected_attn_alignment = self.attn.projected_attn_alignment

            _context_dim = tf.shape(_context)
            _batch_size = _context_dim[0]
            _r_context = \
                tf.reshape(_context, shape=[-1, _max_of_context_tokens, _token_dim])

            # [None, None, _n_hidden]
            _projected_key = \
                tf.layers.dense(_raw_key, _n_hidden, kernel_initializer=xav())

            # [None, None, _n_hidden]
            _projected_context = \
                tf.layers.dense(_r_context, _n_hidden, kernel_initializer=xav())
            _projected_key_dim = tf.shape(_projected_key)
            _r_projected_key = tf.reshape(_projected_key, shape=[-1, _n_hidden, 1])

            _r_projected_key = tf.tile(tf.reshape(
                _projected_key, shape=[-1, 1, _n_hidden]), [1, _max_of_context_tokens, 1])
            _concat_h_state = tf.concat([_projected_context, _r_projected_key], -1)
            _projected_state = tf.layers.dense(_concat_h_state,
                                               _n_hidden, use_bias=False,
                                               kernel_initializer=xav())
            _score = tf.layers.dense(tf.tanh(_projected_state), units=1,
                                     use_bias=False,
                                     kernel_initializer=xav())

            _attn = tf.nn.softmax(_score, dim=1)
            # self.debug_pipe = {'_attn':_attn,'_score':_score}
            if _projected_attn_alignment:
                log.info("Using projected attnention alignment")
                _t_context = tf.transpose(_projected_context, [0, 2, 1])
                _output_tensor = tf.reshape(tf.matmul(_t_context, _attn), shape=[
                                            _batch_size, -1, _n_hidden])
            else:
                log.info("Using without projected attnention alignment")
                _t_context = tf.transpose(_r_context, [0, 2, 1])
                _output_tensor = tf.reshape(tf.matmul(_t_context, _attn), shape=[
                                            _batch_size, -1, _token_dim])
        return _output_tensor

    def _bahdanau_att_mech(self):
        with tf.variable_scope("attention_mechanism/general"):

            _raw_key = self._key
            _n_hidden = (self.attn.att_hidden_dim//2)*2
            _max_of_context_tokens = self.attn.max_of_context_tokens
            _token_dim = self.attn.token_dim
            _context = self._emb_context
            _projected_attn_alignment = self.attn.projected_attn_alignment

            _context_dim = tf.shape(_context)
            _batch_size = _context_dim[0]
            _r_context = \
                tf.reshape(_context, shape=[-1, _max_of_context_tokens, _token_dim])

            # [None, None, _n_hidden]
            _projected_key = \
                tf.layers.dense(_raw_key, _n_hidden, kernel_initializer=xav())
            _projected_key_dim = tf.shape(_projected_key)
            _r_projected_key = \
                tf.tile(tf.reshape(_projected_key, shape=[-1, 1, _n_hidden]),
                        [1, _max_of_context_tokens, 1])

            _lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            _lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            (_output_fw, _output_bw), _states = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=_lstm_fw_cell,
                                                cell_bw=_lstm_bw_cell,
                                                inputs=_r_context,
                                                dtype=tf.float32)

            # [-1,self.max_of_context_tokens,_n_hidden])
            _bilstm_output = tf.concat([_output_fw, _output_bw], -1)
            _concat_h_state = tf.concat([_r_projected_key, _output_fw, _output_bw], -1)
            _projected_state = tf.layers.dense(_concat_h_state,
                                               _n_hidden, use_bias=False,
                                               kernel_initializer=xav())
            _score = tf.layers.dense(tf.tanh(_projected_state), units=1,
                                     use_bias=False,
                                     kernel_initializer=xav())

            _attn = tf.nn.softmax(_score, dim=1)

            # _t_bilstm_output = tf.transpose(_bilstm_output, [0, 2, 1])
            # _output_tensor = tf.reshape(tf.matmul(_t_bilstm_output,_attn),
            #                              shape = [_batch_size, -1, _n_hidden])

            if _projected_attn_alignment:
                log.info("Using projected attnention alignment")
                _t_context = tf.transpose(_bilstm_output, [0, 2, 1])
                _output_tensor = tf.reshape(tf.matmul(_t_context, _attn),
                                            shape=[_batch_size, -1, _n_hidden])
            else:
                log.info("Using without projected attnention alignment")
                _t_context = tf.transpose(_r_context, [0, 2, 1])
                _output_tensor = tf.reshape(tf.matmul(_t_context, _attn),
                                            shape=[_batch_size, -1, _token_dim])
        return _output_tensor

    def _cs_general_att_mech(self):
        with tf.variable_scope("attention_mechanism/cs_general"):

            _raw_key = self._key
            _attn_depth = self.attn.attention_depth
            _n_hidden = (self.attn.att_hidden_dim//2)*2
            _max_of_context_tokens = self.attn.max_of_context_tokens
            _token_dim = self.attn.token_dim
            _key_dim = self.attn.key_dim
            _context = self._emb_context
            _projected_attn_alignment = self.attn.projected_attn_alignment

            _context_dim = tf.shape(_context)
            _batch_size = _context_dim[0]
            _r_context = \
                tf.reshape(_context, shape=[-1, _max_of_context_tokens, _token_dim])
            _r_context = tf.layers.dense(_r_context, _token_dim,
                                         kernel_initializer=xav(), name='r_context')

            # _projected_key = \
            #         tf.layers.dense(_raw_key,
            #                     _n_hidden,
            #                     kernel_initializer=xav(), name = 'projected_key')
            # [None, None, _n_hidden]
            #
            # _projected_key_dim = tf.shape(_projected_key)
            # _r_projected_key = tf.tile(tf.reshape(_projected_key,
            # shape = [-1,1, _n_hidden]), [1,_max_of_context_tokens,1])

            assert _attn_depth is not None

            _lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            _lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            (_output_fw, _output_bw), _states = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=_lstm_fw_cell,
                                                cell_bw=_lstm_bw_cell,
                                                inputs=_r_context,
                                                dtype=tf.float32)
            _bilstm_output = tf.concat([_output_fw, _output_bw], -1)
            _hidden_for_sketch = _bilstm_output
            _key = _raw_key

            if _projected_attn_alignment:
                log.info("Using projected attnention alignment")
                _hidden_for_attn_alignment = _bilstm_output
                _aligned_hidden = csoftmax_attention.attention_gen_block(
                    _hidden_for_sketch, _hidden_for_attn_alignment, _key, _attn_depth)
                _output_tensor = \
                    tf.reshape(_aligned_hidden,
                               shape=[_batch_size, -1, _attn_depth * _n_hidden])
            else:
                log.info("Using without projected attnention alignment")
                _hidden_for_attn_alignment = _r_context
                _aligned_hidden = csoftmax_attention.attention_gen_block(
                    _hidden_for_sketch, _hidden_for_attn_alignment, _key, _attn_depth)
                _output_tensor = \
                    tf.reshape(_aligned_hidden, 
                               shape=[_batch_size, -1, _attn_depth * _token_dim])
        return _output_tensor

    def _cs_bah_att_mech(self):
        with tf.variable_scope("attention_mechanism/cs_bahdanau"):

            _raw_key = self._key
            _attn_depth = self.attn.attention_depth
            _n_hidden = (self.attn.att_hidden_dim//2)*2
            _max_of_context_tokens = self.attn.max_of_context_tokens
            _token_dim = self.attn.token_dim
            _key_dim = self.attn.key_dim
            _context = self._emb_context
            _projected_attn_alignment = self.attn.projected_attn_alignment

            _context_dim = tf.shape(_context)
            _batch_size = _context_dim[0]
            _r_context = \
                tf.reshape(_context, shape=[-1, _max_of_context_tokens, _token_dim])
            _r_context = tf.layers.dense(_r_context,  _token_dim,
                                         kernel_initializer=xav(), name='r_context')

            _projected_key = tf.layers.dense(_raw_key, _n_hidden,
                                             kernel_initializer=xav(),
                                             name='projected_key')
            # [None, None, _n_hidden]

            _projected_key_dim = tf.shape(_projected_key)
            _r_projected_key = \
                tf.tile(tf.reshape(_projected_key, shape=[-1, 1, _n_hidden]),
                        [1, _max_of_context_tokens, 1])

            assert _attn_depth is not None

            _lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            _lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(_n_hidden//2)
            (_output_fw, _output_bw), _states = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=_lstm_fw_cell,
                                                cell_bw=_lstm_bw_cell,
                                                inputs=_r_context,
                                                dtype=tf.float32)
            _bilstm_output = tf.concat([_output_fw, _output_bw], -1)
            _hidden_for_sketch = tf.concat([_r_projected_key, _output_fw, _output_bw], -1)

            if _projected_attn_alignment:
                log.info("Using projected attnention alignment")
                _hidden_for_attn_alignment = _bilstm_output
                _aligned_hidden = csoftmax_attention.attention_bah_block(
                    _hidden_for_sketch, _hidden_for_attn_alignment, _attn_depth)
                _output_tensor = \
                    tf.reshape(_aligned_hidden,
                               shape=[_batch_size, -1, _attn_depth * _n_hidden])
            else:
                log.info("Using without projected attnention alignment")
                _hidden_for_attn_alignment = _r_context
                _aligned_hidden = csoftmax_attention.attention_bah_block(
                    _hidden_for_sketch, _hidden_for_attn_alignment, _attn_depth)
                _output_tensor = \
                    tf.reshape(_aligned_hidden,
                               shape=[_batch_size, -1, _attn_depth * _token_dim])
        return _output_tensor

    def load(self, *args, **kwargs):
        self.load_params()
        super().load(*args, **kwargs)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        path = str(self.save_path.with_suffix('.json').resolve())
        log.info('[saving parameters to {}]'.format(path))
        with open(path, 'w') as fp:
            json.dump(self.opt, fp)

    def load_params(self):
        path = str(self.load_path.with_suffix('.json').resolve())
        log.info('[loading parameters from {}]'.format(path))
        with open(path, 'r') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt[p] != params[p]:
                raise ConfigError("`{}` parameter must be equal to "
                                  "saved model parameter value `{}`"
                                  .format(p, params[p]))

    def reset_state(self):
        # set zero state
        self.state_c = np.zeros([1, self.hidden_size], dtype=np.float32)
        self.state_h = np.zeros([1, self.hidden_size], dtype=np.float32)
 
    def shutdown(self):
        self.sess.close()
