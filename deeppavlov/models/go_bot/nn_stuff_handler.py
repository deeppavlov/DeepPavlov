import collections
import json
from typing import Tuple, List
from logging import getLogger

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.layers import tf_attention_mechanisms as am, tf_layers
from tensorflow.contrib.layers import xavier_initializer as xav

# from deeppavlov.models.go_bot.network import log
# /from deeppavlov.models.go_bot.network import log
from deeppavlov.core.models.tf_model import LRScheduledTFModel


def calc_obs_size(default_tracker_num_features,
                  n_actions,
                  bow_embedder, word_vocab, embedder,
                  intent_classifier, intents):
    obs_size = 6 + default_tracker_num_features + n_actions
    if callable(bow_embedder):
        obs_size += len(word_vocab)
    if callable(embedder):
        obs_size += embedder.dim
    if callable(intent_classifier):
        obs_size += len(intents)
    # log.info(f"Calculated input size for `GoalOrientedBotNetwork` is {obs_size}")
    return obs_size


def configure_attn(curr_attn_token_size,
                   curr_attn_action_as_key,
                   curr_attn_intent_as_key,
                   curr_attn_key_size,
                   embedder,
                   n_actions,
                   intent_classifier,
                   intents):
    token_size = curr_attn_token_size or embedder.dim
    action_as_key = curr_attn_action_as_key or False
    intent_as_key = curr_attn_intent_as_key or False

    possible_key_size = 0
    if action_as_key:
        possible_key_size += n_actions
    if intent_as_key and callable(intent_classifier):
        possible_key_size += len(intents)
    possible_key_size = possible_key_size or 1
    key_size = curr_attn_key_size or possible_key_size

    new_attn = {}
    new_attn['token_size'] = token_size
    new_attn['action_as_key'] = action_as_key
    new_attn['intent_as_key'] = intent_as_key
    new_attn['key_size'] = key_size

    return new_attn

log = getLogger(__name__)


class NNStuffHandler(LRScheduledTFModel):
    SAVE_LOAD_SUBDIR_NAME = "nn_stuff"

    GRAPH_PARAMS = ["hidden_size", "action_size", "dense_size",
                    "attention_mechanism"]
    UNSUPPORTED = ["obs_size"]
    DEPRECATED = ["end_learning_rate", "decay_steps", "decay_power"]

    def train_on_batch(self, x: list, y: list):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __init__(self,
                 hidden_size,
                 action_size,
                 dropout_rate,
                 l2_reg_coef,
                 dense_size,
                 attention_mechanism,
                 network_parameters,
                 embedder,
                 n_actions,
                 intent_classifier,
                 intents,
                 default_tracker_num_features,
                 bow_embedder,
                 word_vocab,
                 load_path,
                 save_path,
                 **kwargs):

        network_parameters = network_parameters or {}
        if any(p in network_parameters for p in self.DEPRECATED):
            log.warning(f"parameters {self.DEPRECATED} are deprecated,"
                        f" for learning rate schedule documentation see"
                        f" deeppavlov.core.models.lr_scheduled_tf_model"
                        f" or read a github tutorial on super convergence.")

        if 'learning_rate' in network_parameters:
            kwargs['learning_rate'] = network_parameters.pop('learning_rate')

        super().__init__(load_path=load_path, save_path=save_path, **kwargs)

        self.configure_network_opts(
            hidden_size,
            action_size,
            dropout_rate,
            l2_reg_coef,
            dense_size,
            attention_mechanism,
            network_parameters,
            embedder,
            n_actions,
            intent_classifier,
            intents,
            default_tracker_num_features,
            bow_embedder,
            word_vocab)


    def _configure_network(self):
        self._init_network_params()
        self._build_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _init_network_params(self) -> None:
        self.dropout_rate = self.opt['dropout_rate']  # todo does dropout actually work
        self.hidden_size = self.opt['hidden_size']
        self.action_size = self.opt['action_size']
        self.dense_size = self.opt['dense_size']
        self.l2_reg = self.opt['l2_reg_coef']

        attn = self.opt.get('attention_mechanism')
        if attn:
            self.attn = collections.namedtuple('attention_mechanism', attn.keys())(**attn)
            self.obs_size -= attn['token_size']
        else:
            self.attn = None

    def _build_graph(self) -> None:
        # todo тут ещё и фичер инжиниринг

        self._add_placeholders()  # todo какая-то тензорфлововая тема

        # build body
        _logits, self._state = self._build_body()

        # probabilities normalization : elemwise multiply with action mask
        _logits_exp = tf.multiply(tf.exp(_logits), self._action_mask)
        _logits_exp_sum = tf.expand_dims(tf.reduce_sum(_logits_exp, -1), -1)
        self._probs = tf.squeeze(_logits_exp / _logits_exp_sum, name='probs')

        # loss, train and predict operations
        self._prediction = tf.argmax(self._probs, axis=-1, name='prediction')

        # _weights = tf.expand_dims(self._utterance_mask, -1)
        # TODO: try multiplying logits to action_mask
        onehots = tf.one_hot(self._action, self.action_size)
        _loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=_logits, labels=onehots
        )
        # multiply with batch utterance mask
        _loss_tensor = tf.multiply(_loss_tensor, self._utterance_mask)
        self._loss = tf.reduce_mean(_loss_tensor, name='loss')
        self._loss += self.l2_reg * tf.losses.get_regularization_loss()
        self._train_op = self.get_train_op(self._loss)

    def _add_placeholders(self) -> None:
        # todo узнай что такое плейсхолдеры в тф
        self._dropout_keep_prob = tf.placeholder_with_default(1.0,
                                                                   shape=[],
                                                                   name='dropout_prob')
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
        self._batch_size = tf.shape(self._features)[0]
        zero_state = tf.zeros([self._batch_size, self.hidden_size], dtype=tf.float32)
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

    def _build_body(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # input projection
        _units = tf.layers.dense(self._features, self.dense_size,
                                 kernel_regularizer=tf.nn.l2_loss,
                                 kernel_initializer=xav())
        if self.attn:
            attn_scope = f"attention_mechanism/{self.attn.type}"
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
        _utter_lengths = tf.cast(tf.reduce_sum(self._utterance_mask, axis=-1),
                                 tf.int32)
        # _output: [batch_size, max_time, hidden_size]
        # _state: tuple of two [batch_size, hidden_size]
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _units,
                                            time_major=False,
                                            initial_state=self._initial_state,
                                            sequence_length=_utter_lengths)
        _output = tf.reshape(_output, (self._batch_size, -1, self.hidden_size))
        _output = tf_layers.variational_dropout(_output,
                                                keep_prob=self._dropout_keep_prob)
        # output projection
        _logits = tf.layers.dense(_output, self.action_size,
                                  kernel_regularizer=tf.nn.l2_loss,
                                  kernel_initializer=xav(), name='logits')
        return _logits, _state

    def configure_network_opts(self,
                               hidden_size,
                               action_size,
                               dropout_rate,
                               l2_reg_coef,
                               dense_size,
                               attention_mechanism,
                               network_parameters,
                               embedder, n_actions, intent_classifier, intents, default_tracker_num_features,
                               bow_embedder, word_vocab) -> None:

        new_network_parameters = {
            'hidden_size': hidden_size,
            'action_size': action_size,
            'dropout_rate': dropout_rate,
            'l2_reg_coef': l2_reg_coef,
            'dense_size': dense_size,
            'attn': attention_mechanism
        }  # network params

        if 'attention_mechanism' in network_parameters:
            # todo чекнуть что всё норм с оригинальными и новыми параметрами сети
            network_parameters['attn'] = network_parameters.pop('attention_mechanism')  # network params
        new_network_parameters.update(network_parameters)  # network params

        hidden_size = new_network_parameters["hidden_size"]
        action_size = new_network_parameters["action_size"]
        dropout_rate = new_network_parameters["dropout_rate"]
        l2_reg_coef = new_network_parameters["l2_reg_coef"]
        dense_size = new_network_parameters["dense_size"]
        attn = new_network_parameters["attn"]

        dense_size = dense_size or hidden_size

        obs_size = calc_obs_size(default_tracker_num_features, n_actions,
                                 bow_embedder, word_vocab, embedder,
                                 intent_classifier, intents)
        if action_size is None:
            action_size = n_actions
        if attn:
            attn.update(configure_attn(curr_attn_token_size=attn.get('token_size'),
                                       curr_attn_action_as_key=attn.get('action_as_key'),
                                       curr_attn_intent_as_key=attn.get('intent_as_key'),
                                       curr_attn_key_size=attn.get('key_size'),
                                       embedder=embedder,
                                       n_actions=n_actions,
                                       intent_classifier=intent_classifier,
                                       intents=intents))
        # specify model options
        opt = {
            'hidden_size': hidden_size,
            'action_size': action_size,
            'dense_size': dense_size,
            'dropout_rate': dropout_rate,
            'l2_reg_coef': l2_reg_coef,
            'attention_mechanism': attn
        }

        self.opt = opt
        self.obs_size = obs_size

        self._configure_network()

    def train_checkpoint_exists(self):
        return tf.train.checkpoint_exists(str(self.load_path.resolve()))

    def load(self, *args, **kwargs) -> None:
        self._load_nn_params()
        super().load(*args, **kwargs)

    def _load_nn_params(self) -> None:
        # todo правда ли что тут загружаются только связанные с нейронкой вещи?

        path = str(self.load_path.with_suffix('.json').resolve())
        # log.info(f"[loading parameters from {path}]")
        with open(path, 'r', encoding='utf8') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt.get(p) != params.get(p):
                raise ConfigError(f"`{p}` parameter must be equal to saved"
                                  f" model parameter value `{params.get(p)}`,"
                                  f" but is equal to `{self.opt.get(p)}`")

    def save(self, *args, **kwargs) -> None:
        super().save(*args, **kwargs)
        self._save_nn_params()

    def _save_nn_params(self) -> None:
        path = str(self.save_path.with_suffix('.json').resolve())
        # log.info(f"[saving parameters to {path}]")
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(self.opt, fp)

    def _network_train_on_batch(self,
                                features: np.ndarray,
                                emb_context: np.ndarray,
                                key: np.ndarray,
                                utter_mask: np.ndarray,
                                action_mask: np.ndarray,
                                action: np.ndarray) -> dict:
        feed_dict = {
            self._dropout_keep_prob: 1.,
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
        return {'loss': loss_value,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum()}


    def _network_call(self, features: np.ndarray, emb_context: np.ndarray, key: np.ndarray,
                      action_mask: np.ndarray, states_c: np.ndarray, states_h: np.ndarray, prob: bool = False) -> List[np.ndarray]:
        feed_dict = {
            self._features: features,
            self._dropout_keep_prob: 1.,
            self._utterance_mask: [[1.]],
            self._initial_state: (states_c, states_h),
            self._action_mask: action_mask
        }
        if self.attn:
            feed_dict[self._emb_context] = emb_context
            feed_dict[self._key] = key

        probs, prediction, state = \
            self.sess.run([self._probs, self._prediction, self._state],
                          feed_dict=feed_dict)

        if prob:
            return probs, state[0], state[1]
        return prediction, state[0], state[1]
