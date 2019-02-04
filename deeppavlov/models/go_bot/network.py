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

import collections
import json
import re
from logging import getLogger
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xav

import deeppavlov.models.go_bot.templates as templ
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.layers import tf_attention_mechanisms as am
from deeppavlov.core.layers import tf_layers
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.lr_scheduled_tf_model import LRScheduledTFModel
from deeppavlov.models.go_bot.tracker import Tracker

log = getLogger(__name__)


@register("go_bot")
class GoalOrientedBot(LRScheduledTFModel):
    """
    The dialogue bot is based on  https://arxiv.org/abs/1702.03274, which introduces
    Hybrid Code Networks that combine an RNN with domain-specific knowledge
    and system action templates.

    The network handles dialogue policy management.
    Inputs features of an utterance and predicts label of a bot action
    (classification task).

    An LSTM with a dense layer for input features and a dense layer for it's output.
    Softmax is used as an output activation function.

    Todo:
        add docstring for trackers.

    Parameters:
        tokenizer: one of tokenizers from
            :doc:`deeppavlov.models.tokenizers </apiref/models/tokenizers>` module.
        tracker: dialogue state tracker from
            :doc:`deeppavlov.models.go_bot.tracker </apiref/models/go_bot>`.
        hidden_size: size of rnn hidden layer.
        action_size: size of rnn output (equals to number of bot actions).
        obs_size: input features' size (must be equal to sum of output sizes of
            ``bow_embedder``, ``embedder``, ``intent_classifier``,
            ``tracker.num_features`` plus size of context features(=6) and
            ``action_size``).
        dropout_rate: probability of weights dropping out.
        l2_reg_coef: l2 regularization weight (applied to input and output layer).
        dense_size: rnn input size.
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
        network_parameters: dictionary with network parameters (for compatibility with release 0.1.1, deprecated in the future)

        template_path: file with mapping between actions and text templates
            for response generation.
        template_type: type of used response templates in string format.
        word_vocab: vocabulary of input word tokens
            (:class:`~deeppavlov.core.data.vocab.DefaultVocabulary` recommended).
        bow_embedder: instance of one-hot word encoder
            :class:`~deeppavlov.models.embedders.bow_embedder.BoWEmbedder`.
        embedder: one of embedders from
            :doc:`deeppavlov.models.embedders </apiref/models/embedders>` module.
        slot_filler: component that outputs slot values for a given utterance
            (:class:`~deeppavlov.models.slotfill.slotfill.DstcSlotFillingNetwork`
            recommended).
        intent_classifier: component that outputs intents probability distribution
            for a given utterance (
            :class:`~deeppavlov.models.classifiers.keras_classification_model.KerasClassificationModel`
            recommended).
        database: database that will be used during inference to perform
            ``api_call_action`` actions and get ``'db_result'`` result (
            :class:`~deeppavlov.core.data.sqlite_database.Sqlite3Database`
            recommended).
        api_call_action: label of the action that corresponds to database api call
            (it must be present in your ``template_path`` file), during interaction
            it will be used to get ``'db_result'`` from ``database``.
        use_action_mask: if ``True``, network output will be applied with a mask
            over allowed actions.
        debug: whether to display debug output.
    """

    GRAPH_PARAMS = ["hidden_size", "action_size", "dense_size", "obs_size",
                    "attention_mechanism"]
    DEPRECATED = ["end_learning_rate", "decay_steps", "decay_power"]

    def __init__(self,
                 tokenizer: Component,
                 tracker: Tracker,
                 template_path: str,
                 save_path: str,
                 hidden_size: int = 128,
                 obs_size: int = None,
                 action_size: int = None,
                 dropout_rate: float = 0.,
                 l2_reg_coef: float = 0.,
                 dense_size: int = None,
                 attention_mechanism: dict = None,
                 network_parameters: Dict[str, Any] = {},
                 load_path: str = None,
                 template_type: str = "DefaultTemplate",
                 word_vocab: Component = None,
                 bow_embedder: Component = None,
                 embedder: Component = None,
                 slot_filler: Component = None,
                 intent_classifier: Component = None,
                 database: Component = None,
                 api_call_action: str = None,  # TODO: make it unrequired
                 use_action_mask: bool = False,
                 debug: bool = False,
                 **kwargs):
        if any(p in network_parameters for p in self.DEPRECATED):
            log.warning(f"parameters {self.DEPRECATED} are deprecated,"
                              " for learning rate schedule documentation see"
                              " deeppavlov.core.models.lr_scheduled_tf_model"
                              " or read gitub tutorial on super convergence.")
        if 'learning_rate' in network_parameters:
            kwargs['learning_rate'] = network_parameters.pop('learning_rate')
        super().__init__(load_path=load_path, save_path=save_path, **kwargs)

        self.tokenizer = tokenizer
        self.tracker = tracker
        self.bow_embedder = bow_embedder
        self.embedder = embedder
        self.slot_filler = slot_filler
        self.intent_classifier = intent_classifier
        self.use_action_mask = use_action_mask
        self.debug = debug
        self.word_vocab = word_vocab

        template_path = expand_path(template_path)
        template_type = getattr(templ, template_type)
        log.info("[loading templates from {}]".format(template_path))
        self.templates = templ.Templates(template_type).load(template_path)
        self.n_actions = len(self.templates)
        log.info("{} templates loaded".format(self.n_actions))

        self.database = database
        self.api_call_id = None
        if api_call_action is not None:
            self.api_call_id = self.templates.actions.index(api_call_action)

        self.intents = []
        if callable(self.intent_classifier):
            self.intents = self.intent_classifier.get_main_component().classes

        new_network_parameters = {
            'hidden_size': hidden_size,
            'action_size': action_size,
            'obs_size': obs_size,
            'dropout_rate': dropout_rate,
            'l2_reg_coef': l2_reg_coef,
            'dense_size': dense_size,
            'attn': attention_mechanism
        }
        if 'attention_mechanism' in network_parameters:
            network_parameters['attn'] = network_parameters.pop('attention_mechanism')
        new_network_parameters.update(network_parameters)
        self._init_network(**new_network_parameters)

        self.reset()

    def _init_network(self, hidden_size, action_size, obs_size, dropout_rate,
                      l2_reg_coef, dense_size, attn):
        # initialize network
        dense_size = dense_size or hidden_size
        if obs_size is None:
            obs_size = 6 + self.tracker.num_features + self.n_actions
            if callable(self.bow_embedder):
                obs_size += len(self.word_vocab)
            if callable(self.embedder):
                obs_size += self.embedder.dim
            if callable(self.intent_classifier):
                obs_size += len(self.intents)
            log.info(f"Calculated input size for `GoalOrientedBotNetwork` is {obs_size}")
        if action_size is None:
            action_size = self.n_actions

        if attn:
            attn['token_size'] = attn.get('token_size') or self.embedder.dim
            attn['action_as_key'] = attn.get('action_as_key', False)
            attn['intent_as_key'] = attn.get('intent_as_key', False)

            key_size = 0
            if attn['action_as_key']:
                key_size += self.n_actions
            if attn['intent_as_key'] and callable(self.intent_classifier):
                key_size += len(self.intents)
            key_size = key_size or 1
            attn['key_size'] = attn.get('key_size') or key_size

        # specify model options
        self.opt = {
            'hidden_size': hidden_size,
            'action_size': action_size,
            'obs_size': obs_size,
            'dense_size': dense_size,
            'dropout_rate': dropout_rate,
            'l2_reg_coef': l2_reg_coef,
            'attention_mechanism': attn
        }

        # initialize parameters
        self._init_network_params()
        # build computational graph
        self._build_graph()
        # initialize session
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        if tf.train.checkpoint_exists(str(self.load_path.resolve())):
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.load()
        else:
            log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))

    def _encode_context(self, context, db_result=None):
        # tokenize input
        tokens = self.tokenizer([context.lower().strip()])[0]
        if self.debug:
            log.debug("Tokenized text= `{}`".format(' '.join(tokens)))

        # Bag of words features
        bow_features = []
        if callable(self.bow_embedder):
            tokens_idx = self.word_vocab(tokens)
            bow_features = self.bow_embedder([tokens_idx])[0]
            bow_features = bow_features.astype(np.float32)

        # Embeddings
        emb_features = []
        emb_context = np.array([], dtype=np.float32)
        if callable(self.embedder):
            if self.attn:
                if tokens:
                    pad = np.zeros((self.attn.max_num_tokens,
                                    self.attn.token_size),
                                   dtype=np.float32)
                    sen = np.array(self.embedder([tokens])[0])
                    # TODO : Unsupport of batch_size more than 1
                    emb_context = np.concatenate((pad, sen))
                    emb_context = emb_context[-self.attn.max_num_tokens:]
                else:
                    emb_context = np.zeros((self.attn.max_num_tokens,
                                            self.attn.token_size),
                                           dtype=np.float32)
            else:
                emb_features = self.embedder([tokens], mean=True)[0]
                # random embedding instead of zeros
                if np.all(emb_features < 1e-20):
                    emb_dim = self.embedder.dim
                    emb_features = np.fabs(np.random.normal(0, 1/emb_dim, emb_dim))

        # Intent features
        intent_features = []
        if callable(self.intent_classifier):
            intent_features = self.intent_classifier([context])[0]

            if self.debug:
                intent = self.intents[np.argmax(intent_features[0])]
                log.debug("Predicted intent = `{}`".format(intent))

        attn_key = np.array([], dtype=np.float32)
        if self.attn:
            if self.attn.action_as_key:
                attn_key = np.hstack((attn_key, self.prev_action))
            if self.attn.intent_as_key:
                attn_key = np.hstack((attn_key, intent_features))
            if len(attn_key) == 0:
                attn_key = np.array([1], dtype=np.float32)

        # Text entity features
        if callable(self.slot_filler):
            self.tracker.update_state(self.slot_filler([tokens])[0])
            if self.debug:
                log.debug("Slot vals: {}".format(self.slot_filler([tokens])))

        state_features = self.tracker.get_features()

        # Other features
        result_matches_state = 0.
        if self.db_result is not None:
            result_matches_state = all(v == self.db_result.get(s)
                                       for s, v in self.tracker.get_state().items()
                                       if v != 'dontcare') * 1.
        context_features = np.array([bool(db_result) * 1.,
                                     (db_result == {}) * 1.,
                                     (self.db_result is None) * 1.,
                                     bool(self.db_result) * 1.,
                                     (self.db_result == {}) * 1.,
                                     result_matches_state],
                                    dtype=np.float32)

        if self.debug:
            log.debug("Context features = {}".format(context_features))
            debug_msg = "num bow features = {}, ".format(len(bow_features)) +\
                        "num emb features = {}, ".format(len(emb_features)) +\
                        "num intent features = {}, ".format(len(intent_features)) +\
                        "num state features = {}, ".format(len(state_features)) +\
                        "num context features = {}, ".format(len(context_features)) +\
                        "prev_action shape = {}".format(len(self.prev_action))
            log.debug(debug_msg)

        concat_feats = np.hstack((bow_features, emb_features, intent_features,
                                  state_features, context_features, self.prev_action))
        return concat_feats, emb_context, attn_key

    def _encode_response(self, act):
        return self.templates.actions.index(act)

    def _decode_response(self, action_id):
        """
        Convert action template id and entities from tracker
        to final response.
        """
        template = self.templates.templates[int(action_id)]

        slots = self.tracker.get_state()
        if self.db_result is not None:
            for k, v in self.db_result.items():
                slots[k] = str(v)

        resp = template.generate_text(slots)
        # in api calls replace unknown slots to "dontcare"
        if (self.templates.ttype is templ.DualTemplate) and\
                (action_id == self.api_call_id):
            resp = re.sub("#([A-Za-z]+)", "dontcare", resp).lower()
        if self.debug:
            log.debug("Pred response = {}".format(resp))
        return resp

    def calc_action_mask(self, previous_action):
        mask = np.ones(self.n_actions, dtype=np.float32)
        if self.use_action_mask:
            known_entities = {**self.tracker.get_state(), **(self.db_result or {})}
            for a_id in range(self.n_actions):
                tmpl = str(self.templates.templates[a_id])
                for entity in set(re.findall('#([A-Za-z]+)', tmpl)):
                    if entity not in known_entities:
                        mask[a_id] = 0.
        # forbid two api calls in a row
        if np.any(previous_action):
            prev_act_id = np.argmax(previous_action)
            if prev_act_id == self.api_call_id:
                mask[prev_act_id] = 0.
        return mask

    def prepare_data(self, x, y):
        b_features, b_u_masks, b_a_masks, b_actions = [], [], [], []
        b_emb_context, b_keys = [], []  # for attention
        max_num_utter = max(len(d_contexts) for d_contexts in x)
        for d_contexts, d_responses in zip(x, y):
            self.reset()
            if self.debug:
                preds = self._infer_dialog(d_contexts)
            d_features, d_a_masks, d_actions = [], [], []
            d_emb_context, d_key = [], []  # for attention
            for context, response in zip(d_contexts, d_responses):
                if context.get('db_result') is not None:
                    self.db_result = context['db_result']
                features, emb_context, key = \
                    self._encode_context(context['text'], context.get('db_result'))
                d_features.append(features)
                d_emb_context.append(emb_context)
                d_key.append(key)
                d_a_masks.append(self.calc_action_mask(self.prev_action))

                action_id = self._encode_response(response['act'])
                d_actions.append(action_id)
                # previous action is teacher-forced here
                self.prev_action *= 0.
                self.prev_action[action_id] = 1.

                if self.debug:
                    log.debug("True response = `{}`".format(response['text']))
                    if preds[0].lower() != response['text'].lower():
                        log.debug("Pred response = `{}`".format(preds[0]))
                    preds = preds[1:]
                    if d_a_masks[-1][action_id] != 1.:
                        log.warn("True action forbidden by action mask.")

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

    def train_on_batch(self, x, y):
        return self.network_train_on_batch(*self.prepare_data(x, y))

    def _infer(self, context, db_result=None, prob=False):
        if db_result is not None:
            self.db_result = db_result
        features, emb_context, key = self._encode_context(context, db_result)
        action_mask = self.calc_action_mask(self.prev_action)
        probs = self.network_call([[features]], [[emb_context]], [[key]],
                                  [[action_mask]], prob=True)
        pred_id = np.argmax(probs)

        # one-hot encoding seems to work better then probabilities
        if prob:
            self.prev_action = probs
        else:
            self.prev_action *= 0
            self.prev_action[pred_id] = 1

        return self._decode_response(pred_id)

    def _infer_dialog(self, contexts):
        self.reset()
        res = []
        for context in contexts:
            if context.get('prev_resp_act') is not None:
                action_id = self._encode_response(context.get('prev_resp_act'))
                # previous action is teacher-forced
                self.prev_action *= 0.
                self.prev_action[action_id] = 1.

            res.append(self._infer(context['text'], db_result=context.get('db_result')))
        return res

    def make_api_call(self, slots):
        db_results = []
        if self.database is not None:
            # filter slot keys with value equal to 'dontcare' as
            # there is no such value in database records
            # and remove unknown slot keys (for example, 'this' in dstc2 tracker)
            db_slots = {s: v for s, v in slots.items()
                        if (v != 'dontcare') and (s in self.database.keys)}
            db_results = self.database([db_slots])[0]
        else:
            log.warn("No database specified.")
        log.info("Made api_call with {}, got {} results.".format(slots, len(db_results)))
        # filter api results if there are more than one
        if len(db_results) > 1:
            db_results = [r for r in db_results if r != self.db_result]
        return db_results[0] if db_results else {}

    def __call__(self, batch):
        if isinstance(batch[0], str):
            res = []
            for x in batch:
                pred = self._infer(x)
                # if made api_call, then respond with next prediction
                prev_act_id = np.argmax(self.prev_action)
                if prev_act_id == self.api_call_id:
                    db_result = self.make_api_call(self.tracker.get_state())
                    res.append(self._infer(x, db_result=db_result))
                else:
                    res.append(pred)
            return res
        return [self._infer_dialog(x) for x in batch]

    def reset(self):
        self.tracker.reset_state()
        self.db_result = None
        self.prev_action = np.zeros(self.n_actions, dtype=np.float32)
        self.reset_network_state()
        if self.debug:
            log.debug("Bot reset.")

    def destroy(self):
        if callable(getattr(self.slot_filler, 'destroy', None)):
            self.slot_filler.destroy()
        if callable(getattr(self.embedder, 'destroy', None)):
            self.embedder.destroy()
        if callable(getattr(self.intent_classifier, 'destroy', None)):
            self.intent_classifier.destroy()
        super().destroy()

    def network_call(self, features, emb_context, key, action_mask, prob=False):
        feed_dict = {
            self._features: features,
            self._dropout_keep_prob: 1.,
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

    def network_train_on_batch(self, features, emb_context, key, utter_mask,
                               action_mask, action):
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
        return {'loss': loss_value}

    def _init_network_params(self):
        self.dropout_rate = self.opt['dropout_rate']
        self.hidden_size = self.opt['hidden_size']
        self.action_size = self.opt['action_size']
        self.obs_size = self.opt['obs_size']
        self.dense_size = self.opt['dense_size']
        self.l2_reg = self.opt['l2_reg_coef']

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

    def _add_placeholders(self):
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
        super().process_event(event_name, data)

    def reset_network_state(self):
        # set zero state
        self.state_c = np.zeros([1, self.hidden_size], dtype=np.float32)
        self.state_h = np.zeros([1, self.hidden_size], dtype=np.float32)
