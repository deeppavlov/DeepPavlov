import collections
from typing import Tuple

import tensorflow as tf

from deeppavlov.core.layers import tf_attention_mechanisms as am, tf_layers
from tensorflow.contrib.layers import xavier_initializer as xav

# from deeppavlov.models.go_bot.network import calc_obs_size, configure_attn



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


class NNStuffHandler():
    def __init__(self):
        pass

    def _init_network_params(self, gobot_obj) -> None:
        gobot_obj.dropout_rate = self.opt['dropout_rate']
        gobot_obj.hidden_size = self.opt['hidden_size']
        gobot_obj.action_size = self.opt['action_size']
        gobot_obj.obs_size = self.opt['obs_size']
        gobot_obj.dense_size = self.opt['dense_size']
        gobot_obj.l2_reg = self.opt['l2_reg_coef']

        attn = self.opt.get('attention_mechanism')
        if attn:
            # gobot_obj.opt['attention_mechanism'] = attn

            gobot_obj.attn = collections.namedtuple('attention_mechanism', attn.keys())(**attn)
            gobot_obj.obs_size -= attn['token_size']
        else:
            gobot_obj.attn = None

    def _build_graph(self, gobot_obj) -> None:
        # todo тут ещё и фичер инжиниринг

        self._add_placeholders(gobot_obj)  # todo какая-то тензорфлововая тема

        # build body
        _logits, gobot_obj._state = self._build_body(gobot_obj)

        # probabilities normalization : elemwise multiply with action mask
        _logits_exp = tf.multiply(tf.exp(_logits), gobot_obj._action_mask)
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
        gobot_obj._train_op = gobot_obj.get_train_op(gobot_obj._loss)

    def _add_placeholders(self, gobot_obj) -> None:
        # todo узнай что такое плейсхолдеры в тф
        gobot_obj._dropout_keep_prob = tf.placeholder_with_default(1.0,
                                                                   shape=[],
                                                                   name='dropout_prob')
        gobot_obj._features = tf.placeholder(tf.float32,
                                             [None, None, gobot_obj.obs_size],
                                             name='features')
        gobot_obj._action = tf.placeholder(tf.int32,
                                           [None, None],
                                           name='ground_truth_action')
        gobot_obj._action_mask = tf.placeholder(tf.float32,
                                                [None, None, gobot_obj.action_size],
                                                name='action_mask')
        gobot_obj._utterance_mask = tf.placeholder(tf.float32,
                                                   shape=[None, None],
                                                   name='utterance_mask')
        gobot_obj._batch_size = tf.shape(gobot_obj._features)[0]
        zero_state = tf.zeros([gobot_obj._batch_size, gobot_obj.hidden_size], dtype=tf.float32)
        _initial_state_c = \
            tf.placeholder_with_default(zero_state, shape=[None, gobot_obj.hidden_size])
        _initial_state_h = \
            tf.placeholder_with_default(zero_state, shape=[None, gobot_obj.hidden_size])
        gobot_obj._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c,
                                                                 _initial_state_h)
        if gobot_obj.attn:
            _emb_context_shape = \
                [None, None, gobot_obj.attn.max_num_tokens, gobot_obj.attn.token_size]
            gobot_obj._emb_context = tf.placeholder(tf.float32,
                                                    _emb_context_shape,
                                                    name='emb_context')
            gobot_obj._key = tf.placeholder(tf.float32,
                                            [None, None, gobot_obj.attn.key_size],
                                            name='key')

    def _build_body(self, gobot_obj) -> Tuple[tf.Tensor, tf.Tensor]:
        # input projection
        _units = tf.layers.dense(gobot_obj._features, gobot_obj.dense_size,
                                 kernel_regularizer=tf.nn.l2_loss,
                                 kernel_initializer=xav())
        if gobot_obj.attn:
            attn_scope = f"attention_mechanism/{gobot_obj.attn.type}"
            with tf.variable_scope(attn_scope):
                if gobot_obj.attn.type == 'general':
                    _attn_output = am.general_attention(
                        gobot_obj._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'bahdanau':
                    _attn_output = am.bahdanau_attention(
                        gobot_obj._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'cs_general':
                    _attn_output = am.cs_general_attention(
                        gobot_obj._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        depth=gobot_obj.attn.depth,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'cs_bahdanau':
                    _attn_output = am.cs_bahdanau_attention(
                        gobot_obj._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        depth=gobot_obj.attn.depth,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'light_general':
                    _attn_output = am.light_general_attention(
                        gobot_obj._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        projected_align=gobot_obj.attn.projected_align)
                elif gobot_obj.attn.type == 'light_bahdanau':
                    _attn_output = am.light_bahdanau_attention(
                        gobot_obj._key,
                        gobot_obj._emb_context,
                        hidden_size=gobot_obj.attn.hidden_size,
                        projected_align=gobot_obj.attn.projected_align)
                else:
                    raise ValueError("wrong value for attention mechanism type")
            _units = tf.concat([_units, _attn_output], -1)

        _units = tf_layers.variational_dropout(_units,
                                               keep_prob=gobot_obj._dropout_keep_prob)

        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(gobot_obj.hidden_size)
        _utter_lengths = tf.cast(tf.reduce_sum(gobot_obj._utterance_mask, axis=-1),
                                 tf.int32)
        # _output: [batch_size, max_time, hidden_size]
        # _state: tuple of two [batch_size, hidden_size]
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell,
                                            _units,
                                            time_major=False,
                                            initial_state=gobot_obj._initial_state,
                                            sequence_length=_utter_lengths)
        _output = tf.reshape(_output, (gobot_obj._batch_size, -1, gobot_obj.hidden_size))
        _output = tf_layers.variational_dropout(_output,
                                                keep_prob=gobot_obj._dropout_keep_prob)
        # output projection
        _logits = tf.layers.dense(_output, gobot_obj.action_size,
                                  kernel_regularizer=tf.nn.l2_loss,
                                  kernel_initializer=xav(), name='logits')
        return _logits, _state


    def configure_network_opts(self, action_size, attn, dense_size, dropout_rate, hidden_size, l2_reg_coef, obs_size,
                               embedder,
                               n_actions,
                               intent_classifier,
                               intents,
                               default_tracker_num_features,
                               bow_embedder,
                               word_vocab):

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
        return opt