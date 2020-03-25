import collections
import json
from typing import Tuple, List

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


class NNStuffHandler(LRScheduledTFModel):
    SAVE_LOAD_SUBDIR_NAME = "nn_stuff"

    GRAPH_PARAMS = ["hidden_size", "action_size", "dense_size", "obs_size",
                    "attention_mechanism"]
    DEPRECATED = ["end_learning_rate", "decay_steps", "decay_power"]

    def train_on_batch(self, x: list, y: list):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __init__(self, load_path, save_path, **kwargs):
        super().__init__(load_path=load_path, save_path=save_path, **kwargs)

    def _configure_network(self, gobot_obj):
        self._init_network_params(gobot_obj)
        self._build_graph(gobot_obj)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _init_network_params(self, gobot_obj) -> None:
        self.dropout_rate = self.opt['dropout_rate']  # todo does dropout actually work
        self.hidden_size = self.opt['hidden_size']
        gobot_obj.action_size = self.opt['action_size']
        self.obs_size = self.opt['obs_size']  # todo что такое обс сайз
        self.dense_size = self.opt['dense_size']
        gobot_obj.l2_reg = self.opt['l2_reg_coef']

        attn = self.opt.get('attention_mechanism')
        if attn:
            # gobot_obj.opt['attention_mechanism'] = attn

            gobot_obj.attn = collections.namedtuple('attention_mechanism', attn.keys())(**attn)
            self.obs_size -= attn['token_size']
        else:
            gobot_obj.attn = None

    def _build_graph(self, gobot_obj) -> None:
        # todo тут ещё и фичер инжиниринг

        self._add_placeholders(gobot_obj)  # todo какая-то тензорфлововая тема

        # build body
        _logits, gobot_obj._state = self._build_body(gobot_obj)

        # probabilities normalization : elemwise multiply with action mask
        _logits_exp = tf.multiply(tf.exp(_logits), self._action_mask)
        _logits_exp_sum = tf.expand_dims(tf.reduce_sum(_logits_exp, -1), -1)
        gobot_obj._probs = tf.squeeze(_logits_exp / _logits_exp_sum, name='probs')

        # loss, train and predict operations
        gobot_obj._prediction = tf.argmax(gobot_obj._probs, axis=-1, name='prediction')

        # _weights = tf.expand_dims(self._utterance_mask, -1)
        # TODO: try multiplying logits to action_mask
        onehots = tf.one_hot(gobot_obj._action, gobot_obj.action_size)
        _loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=_logits, labels=onehots
        )
        # multiply with batch utterance mask
        _loss_tensor = tf.multiply(_loss_tensor, gobot_obj._utterance_mask)
        gobot_obj._loss = tf.reduce_mean(_loss_tensor, name='loss')
        gobot_obj._loss += gobot_obj.l2_reg * tf.losses.get_regularization_loss()
        gobot_obj._train_op = self.get_train_op(gobot_obj._loss)

    def _add_placeholders(self, gobot_obj) -> None:
        # todo узнай что такое плейсхолдеры в тф
        self._dropout_keep_prob = tf.placeholder_with_default(1.0,
                                                                   shape=[],
                                                                   name='dropout_prob')
        self._features = tf.placeholder(tf.float32,
                                             [None, None, self.obs_size],
                                             name='features')
        gobot_obj._action = tf.placeholder(tf.int32,
                                           [None, None],
                                           name='ground_truth_action')
        self._action_mask = tf.placeholder(tf.float32,
                                                [None, None, gobot_obj.action_size],
                                                name='action_mask')
        gobot_obj._utterance_mask = tf.placeholder(tf.float32,
                                                   shape=[None, None],
                                                   name='utterance_mask')
        gobot_obj._batch_size = tf.shape(self._features)[0]
        zero_state = tf.zeros([gobot_obj._batch_size, self.hidden_size], dtype=tf.float32)
        _initial_state_c = \
            tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])
        _initial_state_h = \
            tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])
        gobot_obj._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c,
                                                                 _initial_state_h)
        if gobot_obj.attn:
            _emb_context_shape = \
                [None, None, gobot_obj.attn.max_num_tokens, gobot_obj.attn.token_size]
            gobot_obj._emb_context = tf.placeholder(tf.float32,
                                                    _emb_context_shape,
                                                    name='emb_context')
            self._key = tf.placeholder(tf.float32,
                                            [None, None, gobot_obj.attn.key_size],
                                            name='key')

    def _build_body(self, gobot_obj) -> Tuple[tf.Tensor, tf.Tensor]:
        # input projection
        _units = tf.layers.dense(self._features, self.dense_size,
                                 kernel_regularizer=tf.nn.l2_loss,
                                 kernel_initializer=xav())
        if gobot_obj.attn:
            attn_scope = f"attention_mechanism/{gobot_obj.attn.type}"
            with tf.variable_scope(attn_scope):
                if gobot_obj.attn.type == 'general':
                    _attn_output = am.general_attention(
                        self._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'bahdanau':
                    _attn_output = am.bahdanau_attention(
                        self._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'cs_general':
                    _attn_output = am.cs_general_attention(
                        self._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        depth=gobot_obj.attn.depth,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'cs_bahdanau':
                    _attn_output = am.cs_bahdanau_attention(
                        self._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        depth=gobot_obj.attn.depth,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'light_general':
                    _attn_output = am.light_general_attention(
                        self._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'light_bahdanau':
                    _attn_output = am.light_bahdanau_attention(
                        self._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        projected_align=gobot_obj.attn.projected_align)
                else:
                    raise ValueError("wrong value for attention mechanism type")
            _units = tf.concat([_units, _attn_output], -1)

        _units = tf_layers.variational_dropout(_units,
                                               keep_prob=self._dropout_keep_prob)

        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        _utter_lengths = tf.cast(tf.reduce_sum(gobot_obj._utterance_mask, axis=-1),
                                 tf.int32)
        # _output: [batch_size, max_time, hidden_size]
        # _state: tuple of two [batch_size, hidden_size]
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _units,
                                            time_major=False,
                                            initial_state=gobot_obj._initial_state,
                                            sequence_length=_utter_lengths)
        _output = tf.reshape(_output, (gobot_obj._batch_size, -1, self.hidden_size))
        _output = tf_layers.variational_dropout(_output,
                                                keep_prob=self._dropout_keep_prob)
        # output projection
        _logits = tf.layers.dense(_output, gobot_obj.action_size,
                                  kernel_regularizer=tf.nn.l2_loss,
                                  kernel_initializer=xav(), name='logits')
        return _logits, _state

    def configure_network_opts(self,
                               network_parameters, new_network_parameters,
                               embedder, n_actions, intent_classifier, intents, default_tracker_num_features,
                               bow_embedder, word_vocab) -> None:

        if 'attention_mechanism' in network_parameters:
            network_parameters['attn'] = network_parameters.pop('attention_mechanism')  # network params
        new_network_parameters.update(network_parameters)  # network params

        hidden_size = new_network_parameters["hidden_size"]
        action_size = new_network_parameters["action_size"]
        obs_size = new_network_parameters["obs_size"]
        dropout_rate = new_network_parameters["dropout_rate"]
        l2_reg_coef = new_network_parameters["l2_reg_coef"]
        dense_size = new_network_parameters["dense_size"]
        attn = new_network_parameters["attn"]

        dense_size = dense_size or hidden_size

        if obs_size is None:
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
            'obs_size': obs_size,
            'dense_size': dense_size,
            'dropout_rate': dropout_rate,
            'l2_reg_coef': l2_reg_coef,
            'attention_mechanism': attn
        }

        self.opt = opt

    def train_checkpoint_exists(self, load_path):
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

    def _network_train_on_batch(self, gobot_obj,
                                features: np.ndarray,
                                emb_context: np.ndarray,
                                key: np.ndarray,
                                utter_mask: np.ndarray,
                                action_mask: np.ndarray,
                                action: np.ndarray) -> dict:
        feed_dict = {
            self._dropout_keep_prob: 1.,
            gobot_obj._utterance_mask: utter_mask,
            self._features: features,
            gobot_obj._action: action,
            self._action_mask: action_mask
        }
        if gobot_obj.attn:
            feed_dict[gobot_obj._emb_context] = emb_context
            feed_dict[self._key] = key

        _, loss_value, prediction = \
            self.sess.run([gobot_obj._train_op, gobot_obj._loss, gobot_obj._prediction],
                               feed_dict=feed_dict)
        return {'loss': loss_value,
                'learning_rate': gobot_obj.nn_stuff_handler.get_learning_rate(),
                'momentum': gobot_obj.nn_stuff_handler.get_momentum()}


    def _network_call(self, gobot_obj, features: np.ndarray, emb_context: np.ndarray, key: np.ndarray,
                      action_mask: np.ndarray, states_c: np.ndarray, states_h: np.ndarray, prob: bool = False) -> List[np.ndarray]:
        feed_dict = {
            self._features: features,
            self._dropout_keep_prob: 1.,
            gobot_obj._utterance_mask: [[1.]],
            gobot_obj._initial_state: (states_c, states_h),
            self._action_mask: action_mask
        }
        if gobot_obj.attn:
            feed_dict[gobot_obj._emb_context] = emb_context
            feed_dict[self._key] = key

        probs, prediction, state = \
            self.sess.run([gobot_obj._probs, gobot_obj._prediction, gobot_obj._state],
                          feed_dict=feed_dict)

        if prob:
            return probs, state[0], state[1]
        return prediction, state[0], state[1]

    def _prepare_data(self, gobot_obj, x: List[dict], y: List[dict]) -> List[np.ndarray]:
        b_features, b_u_masks, b_a_masks, b_actions = [], [], [], []
        b_emb_context, b_keys = [], []  # for attention
        max_num_utter = max(len(d_contexts) for d_contexts in x)
        for d_contexts, d_responses in zip(x, y):
            gobot_obj.dialogue_state_tracker.reset_state()
            d_features, d_a_masks, d_actions = [], [], []
            d_emb_context, d_key = [], []  # for attention

            for context, response in zip(d_contexts, d_responses):
                tokens = gobot_obj.tokenizer([context['text'].lower().strip()])[0]

                # update state
                gobot_obj.dialogue_state_tracker.get_ground_truth_db_result_from(context)

                if callable(gobot_obj.slot_filler):
                    context_slots = gobot_obj.slot_filler([tokens])[0]
                    gobot_obj.dialogue_state_tracker.update_state(context_slots)

                features, emb_context, key = gobot_obj.data_handler._encode_context(gobot_obj, tokens,
                                                                                    tracker=gobot_obj.dialogue_state_tracker)
                d_features.append(features)
                d_emb_context.append(emb_context)
                d_key.append(key)
                d_a_masks.append(gobot_obj.dialogue_state_tracker.calc_action_mask(gobot_obj.api_call_id))

                action_id = gobot_obj.data_handler._encode_response(gobot_obj, response['act'])
                d_actions.append(action_id)
                # update state
                # - previous action is teacher-forced here
                gobot_obj.dialogue_state_tracker.update_previous_action(action_id)

                if gobot_obj.debug:
                    # log.debug(f"True response = '{response['text']}'.")
                    if d_a_masks[-1][action_id] != 1.:
                        pass
                        # log.warning("True action forbidden by action mask.")

            # padding to max_num_utter
            num_padds = max_num_utter - len(d_contexts)
            d_features.extend([np.zeros_like(d_features[0])] * num_padds)
            d_emb_context.extend([np.zeros_like(d_emb_context[0])] * num_padds)
            d_key.extend([np.zeros_like(d_key[0])] * num_padds)
            d_u_mask = [1] * len(d_contexts) + [0] * num_padds
            d_a_masks.extend([np.zeros_like(d_a_masks[0])] * num_padds)
            d_actions.extend([0] * num_padds)

            b_features.append(d_features)
            b_emb_context.append(d_emb_context)
            b_keys.append(d_key)
            b_u_masks.append(d_u_mask)
            b_a_masks.append(d_a_masks)
            b_actions.append(d_actions)
        return b_features, b_emb_context, b_keys, b_u_masks, b_a_masks, b_actions
