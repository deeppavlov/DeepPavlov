import json
from typing import Tuple, Optional, Sequence, NamedTuple
from logging import getLogger

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.layers import tf_attention_mechanisms as am, tf_layers

# noinspection PyUnresolvedReferences
from tensorflow.contrib.layers import xavier_initializer as xav

from deeppavlov.core.models.tf_model import LRScheduledTFModel

from deeppavlov.models.go_bot.tokens_vectorizer import TokensVectorRepresentationParams
from deeppavlov.models.go_bot.dto.dataset_features import BatchDialoguesFeatures, BatchDialoguesTargets

# todo
from deeppavlov.models.go_bot.shared_gobot_params import SharedGoBotParams

log = getLogger(__name__)


class PolicyNetworkParams:
    """
    The class to deal with the overcomplicated structure of the GO-bot configs.
    It is initialized from the config-as-is and performs all the conflicting parameters resolution internally.
    """
    # todo remove the complex config logic
    UNSUPPORTED = ["obs_size"]
    DEPRECATED = ["end_learning_rate", "decay_steps", "decay_power"]

    def __init__(self,
                 hidden_size,
                 action_size,
                 dropout_rate,
                 l2_reg_coef,
                 dense_size,
                 attention_mechanism,
                 network_parameters):
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.dropout_rate = dropout_rate
        self.l2_reg_coef = l2_reg_coef
        self.dense_size = dense_size
        self.attention_mechanism = attention_mechanism
        self.network_parameters = network_parameters or {}

        self.log_deprecated_params(self.network_parameters.keys())

    def get_hidden_size(self):
        return self.network_parameters.get("hidden_size", self.hidden_size)

    def get_action_size(self):
        return self.network_parameters.get("action_size", self.action_size)

    def get_dropout_rate(self):
        return self.network_parameters.get("dropout_rate", self.dropout_rate)

    def get_l2_reg_coef(self):
        return self.network_parameters.get("l2_reg_coef", self.l2_reg_coef)

    def get_dense_size(self):
        return self.network_parameters.get("dense_size", self.dense_size) or self.hidden_size

    def get_learning_rate(self):
        return self.network_parameters.get("learning_rate", None)

    def get_attn_params(self):
        return self.network_parameters.get('attention_mechanism', self.attention_mechanism)

    def log_deprecated_params(self, network_parameters):
        if any(p in network_parameters for p in self.DEPRECATED):
            log.warning(f"parameters {self.DEPRECATED} are deprecated,"
                        f" for learning rate schedule documentation see"
                        f" deeppavlov.core.models.lr_scheduled_tf_model"
                        f" or read a github tutorial on super convergence.")


class GobotAttnParams(NamedTuple):
    """
    the DTO-like class that stores the attention mechanism configuration params.
    """
    max_num_tokens: int
    hidden_size: int
    token_size: int
    key_size: int
    type_: str
    projected_align: bool
    depth: int
    action_as_key: bool
    intent_as_key: bool


class PolicyNetwork(LRScheduledTFModel):
    """
    the Policy Network is a ML model whose goal is to choose the right system response when in dialogue with user.
    """

    GRAPH_PARAMS = ["hidden_size", "action_size", "dense_size", "attention_params"]
    SERIALIZABLE_FIELDS = ["hidden_size", "action_size", "dense_size", "dropout_rate", "l2_reg_coef",
                           "attention_params"]

    def __init__(self, network_params_passed: PolicyNetworkParams,
                 tokens_dims: TokensVectorRepresentationParams,
                 features_params: SharedGoBotParams,
                 load_path,
                 save_path,
                 debug=False,
                 **kwargs):
        self.debug = debug
        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"network_params_passed={network_params_passed}, tokens_dims={tokens_dims}, "
                      f"features_params={features_params}, load_path={load_path}, save_path={save_path}, "
                      f"debug={debug}, kwargs={kwargs}")
        if network_params_passed.get_learning_rate():
            kwargs['learning_rate'] = network_params_passed.get_learning_rate()  # todo :(

        super().__init__(load_path=load_path, save_path=save_path, **kwargs)

        self.hidden_size = network_params_passed.get_hidden_size()
        self.action_size = network_params_passed.get_action_size() or features_params.num_actions  # todo :(
        self.dropout_rate = network_params_passed.get_dropout_rate()
        self.l2_reg_coef = network_params_passed.get_l2_reg_coef()
        self.dense_size = network_params_passed.get_dense_size()

        attn_params_passed = network_params_passed.get_attn_params()
        self.attention_params = self.configure_attn(attn_params_passed, tokens_dims, features_params)  # todo :(
        self.input_size = self.calc_input_size(tokens_dims, features_params, self.attention_params)  # todo :(

        if self.debug:
            log.debug(f"INSIDE {self.__class__.__name__} init(). calculated NN hyperparams: "
                      f"attention_params={self.attention_params}, "
                      f"hidden_size={self.hidden_size}, action_size={self.action_size}, "
                      f"dropout_rate={self.dropout_rate}, l2_reg_coef={self.l2_reg_coef}, "
                      f"dense_size={self.dense_size}, input_size={self.input_size}")

        self._build_graph()
        if self.debug:
            log.debug(f"INSIDE {self.__class__.__name__} init(). build graph done.")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.debug:
            log.debug(f"INSIDE {self.__class__.__name__} init(). "
                      f"Session() initialization and global_variables_initializer() done.")

        if self.train_checkpoint_exists():
            log.info(f"INSIDE {self.__class__.__name__} init(). Initializing {self.__class__.__name__} from checkpoint.")
            self.load()
        else:
            log.info(f"INSIDE {self.__class__.__name__} init(). Initializing {self.__class__.__name__} from scratch.")

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} init(): "
                      f"network_params_passed={network_params_passed}, tokens_dims={tokens_dims}, "
                      f"features_params={features_params}, load_path={load_path}, save_path={save_path}, "
                      f"debug={debug}, kwargs={kwargs}")

    @staticmethod
    def calc_input_size(tokens_dims: TokensVectorRepresentationParams,
                        shared_go_bot_params: SharedGoBotParams,
                        attention_params: Optional[GobotAttnParams]) -> int:
        """
        :param tokens_dims: the tokens vectors dimensions
        :param shared_go_bot_params: GO-bot hyperparams used in various parts of the pipeline
        :param attention_params: the params of attention mechanism of the network for which input size is calculated
        :return: the calculated input shape of policy network
        """
        input_size = 6 + shared_go_bot_params.num_tracker_features + shared_go_bot_params.num_actions  # todo: why 6
        if tokens_dims.bow_dim:
            input_size += tokens_dims.bow_dim
        if tokens_dims.embedding_dim:
            input_size += tokens_dims.embedding_dim
        if shared_go_bot_params.num_intents:
            input_size += shared_go_bot_params.num_intents
        if attention_params is not None:
            input_size -= attention_params.token_size

        return input_size

    @staticmethod
    def configure_attn(attn: dict,
                       tokens_dims: TokensVectorRepresentationParams,
                       features_params: SharedGoBotParams):
        # todo store params in proper class objects not in dicts, requires serialization logic update

        if not attn:
            return None

        token_size = tokens_dims.embedding_dim  # todo sync with nn params
        action_as_key = attn.get('action_as_key', False)
        intent_as_key = attn.get('intent_as_key', False)
        key_size = PolicyNetwork.calc_attn_key_size(features_params, action_as_key, intent_as_key)

        gobot_attn_params = GobotAttnParams(max_num_tokens=attn.get("max_num_tokens"),
                                            hidden_size=attn.get("hidden_size"),
                                            token_size=token_size,
                                            key_size=key_size,
                                            type_=attn.get("type"),
                                            projected_align=attn.get("projected_align"),
                                            depth=attn.get("depth"),
                                            action_as_key=action_as_key,
                                            intent_as_key=intent_as_key)

        return gobot_attn_params

    @staticmethod
    def calc_attn_key_size(shared_go_bot_params: SharedGoBotParams, action_as_key: bool, intent_as_key: bool) -> int:
        """
        :param shared_go_bot_params: GO-bot hyperparams used in various parts of the pipeline
        :param action_as_key: True if actions are part of attention keys
        :param intent_as_key: True if intents are part of attention keys
        :return: the calculated attention key shape of policy network
        """
        # True if actions are part of attention keys -- actually *the last predicted action*

        possible_key_size = 0
        if action_as_key:
            possible_key_size += shared_go_bot_params.num_actions
        if intent_as_key and shared_go_bot_params.num_intents:
            possible_key_size += shared_go_bot_params.num_intents
        possible_key_size = possible_key_size or 1  # todo rewrite
        return possible_key_size

    def calc_attn_key(self, intent_features, tracker_prev_action):
        """
        :param intent_features: output intent extractors
        :param tracker_prev_action: one-hot-encoded previous executed action
        :return: vector representing an attention key
        """
        # todo dto-like class for the attn features?

        attn_key = np.array([], dtype=np.float32)

        if self.attention_params:
            if self.attention_params.action_as_key:
                attn_key = np.hstack((attn_key, tracker_prev_action))
            if self.attention_params.intent_as_key:
                attn_key = np.hstack((attn_key, intent_features))
            if len(attn_key) == 0:
                attn_key = np.array([1], dtype=np.float32)
        return attn_key

    def _build_graph(self) -> None:
        self._add_placeholders()

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
        self._loss += self.l2_reg_coef * tf.losses.get_regularization_loss()
        self._train_op = self.get_train_op(self._loss)

    def _add_placeholders(self) -> None:
        self._dropout_keep_prob = tf.placeholder_with_default(1.0, shape=[], name='dropout_prob')

        self._features = tf.placeholder(tf.float32, [None, None, self.input_size], name='features')

        self._action = tf.placeholder(tf.int32, [None, None], name='ground_truth_action')

        self._action_mask = tf.placeholder(tf.float32, [None, None, self.action_size], name='action_mask')

        self._utterance_mask = tf.placeholder(tf.float32, shape=[None, None], name='utterance_mask')

        self._batch_size = tf.shape(self._features)[0]

        zero_state = tf.zeros([self._batch_size, self.hidden_size], dtype=tf.float32)
        _initial_state_c = tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])
        _initial_state_h = tf.placeholder_with_default(zero_state, shape=[None, self.hidden_size])
        self._initial_state = tf.nn.rnn_cell.LSTMStateTuple(_initial_state_c, _initial_state_h)

        if self.attention_params:
            _emb_context_shape = [None, None, self.attention_params.max_num_tokens,
                                  self.attention_params.token_size]
            self._emb_context = tf.placeholder(tf.float32, _emb_context_shape, name='emb_context')
            self._key = tf.placeholder(tf.float32, [None, None, self.attention_params.key_size], name='key')

    def _build_body(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # input projection
        _units = tf.layers.dense(self._features, self.dense_size,
                                 kernel_regularizer=tf.nn.l2_loss, kernel_initializer=xav())

        if self.attention_params:
            _attn_output = self._build_attn_body()
            _units = tf.concat([_units, _attn_output], -1)

        _units = tf_layers.variational_dropout(_units, keep_prob=self._dropout_keep_prob)

        # recurrent network unit
        _lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        _utter_lengths = tf.cast(tf.reduce_sum(self._utterance_mask, axis=-1), tf.int32)

        # _output: [batch_size, max_time, hidden_size]
        # _state: tuple of two [batch_size, hidden_size]
        _output, _state = tf.nn.dynamic_rnn(_lstm_cell, _units,
                                            time_major=False, initial_state=self._initial_state,
                                            sequence_length=_utter_lengths)

        _output = tf.reshape(_output, (self._batch_size, -1, self.hidden_size))
        _output = tf_layers.variational_dropout(_output, keep_prob=self._dropout_keep_prob)
        # output projection
        _logits = tf.layers.dense(_output, self.action_size,
                                  kernel_regularizer=tf.nn.l2_loss, kernel_initializer=xav(), name='logits')
        return _logits, _state

    def _build_attn_body(self):
        attn_scope = f"attention_params/{self.attention_params.type_}"
        with tf.variable_scope(attn_scope):
            if self.attention_params.type_ == 'general':
                _attn_output = am.general_attention(self._key, self._emb_context,
                                                    hidden_size=self.attention_params.hidden_size,
                                                    projected_align=self.attention_params.projected_align)
            elif self.attention_params.type_ == 'bahdanau':
                _attn_output = am.bahdanau_attention(self._key, self._emb_context,
                                                     hidden_size=self.attention_params.hidden_size,
                                                     projected_align=self.attention_params.projected_align)
            elif self.attention_params.type_ == 'cs_general':
                _attn_output = am.cs_general_attention(self._key, self._emb_context,
                                                       hidden_size=self.attention_params.hidden_size,
                                                       depth=self.attention_params.depth,
                                                       projected_align=self.attention_params.projected_align)
            elif self.attention_params.type_ == 'cs_bahdanau':
                _attn_output = am.cs_bahdanau_attention(self._key, self._emb_context,
                                                        hidden_size=self.attention_params.hidden_size,
                                                        depth=self.attention_params.depth,
                                                        projected_align=self.attention_params.projected_align)
            elif self.attention_params.type_ == 'light_general':
                _attn_output = am.light_general_attention(self._key, self._emb_context,
                                                          hidden_size=self.attention_params.hidden_size,
                                                          projected_align=self.attention_params.projected_align)
            elif self.attention_params.type_ == 'light_bahdanau':
                _attn_output = am.light_bahdanau_attention(self._key, self._emb_context,
                                                           hidden_size=self.attention_params.hidden_size,
                                                           projected_align=self.attention_params.projected_align)
            else:
                raise ValueError("wrong value for attention mechanism type")
        return _attn_output

    def train_checkpoint_exists(self):
        return tf.train.checkpoint_exists(str(self.load_path.resolve()))

    def get_attn_hyperparams(self) -> Optional[GobotAttnParams]:
        attn_hyperparams = None
        if self.attention_params:
            attn_hyperparams = self.attention_params
        return attn_hyperparams

    def has_attn(self):
        """:returns: True if the model has an attention mechanism"""
        return self.attention_params is not None

    def get_attn_window_size(self):
        """
        :returns: the length of the window the model looks with attn if the attention mechanism is configured.
                  if the model has no attention mechanism returns None.
        """
        return self.attention_params.max_num_tokens if self.has_attn() else None

    def __call__(self, batch_dialogues_features: BatchDialoguesFeatures,
                 states_c: np.ndarray, states_h: np.ndarray, prob: bool = False,
                 *args, **kwargs) -> Sequence[np.ndarray]:

        states_c = [[states_c]]  # list of list aka batch of dialogues
        states_h = [[states_h]]  # list of list aka batch of dialogues

        feed_dict = {
            self._dropout_keep_prob: 1.,
            self._initial_state: (states_c, states_h),
            self._utterance_mask: batch_dialogues_features.b_padded_dialogue_length_mask,
            self._features: batch_dialogues_features.b_featuress,
            self._action_mask: batch_dialogues_features.b_action_masks
        }
        if self.attention_params:
            feed_dict[self._emb_context] = batch_dialogues_features.b_tokens_embeddings_paddeds
            feed_dict[self._key] = batch_dialogues_features.b_attn_keys

        probs, prediction, state = self.sess.run([self._probs, self._prediction, self._state], feed_dict=feed_dict)

        if prob:
            return probs, state[0], state[1]
        return prediction, state[0], state[1]

    def train_on_batch(self,
                       batch_dialogues_features: BatchDialoguesFeatures,
                       batch_dialogues_targets: BatchDialoguesTargets) -> dict:

        feed_dict = {
            self._dropout_keep_prob: 1.,
            self._utterance_mask: batch_dialogues_features.b_padded_dialogue_length_mask,
            self._features: batch_dialogues_features.b_featuress,
            self._action: batch_dialogues_targets.b_action_ids,
            self._action_mask: batch_dialogues_features.b_action_masks
        }

        if self.attention_params:
            feed_dict[self._emb_context] = batch_dialogues_features.b_tokens_embeddings_paddeds
            feed_dict[self._key] = batch_dialogues_features.b_attn_keys

        _, loss_value, prediction = self.sess.run([self._train_op, self._loss, self._prediction], feed_dict=feed_dict)

        return {'loss': loss_value,
                'learning_rate': self.get_learning_rate(),
                'momentum': self.get_momentum()}

    def load(self, *args, **kwargs) -> None:
        # todo move load_nn_params here?
        self._load_nn_params()
        super().load(*args, **kwargs)

    def _load_nn_params(self) -> None:
        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} _load_nn_params()")

        path = str(self.load_path.with_suffix('.json').resolve())

        if self.debug:
            log.debug(f"INSIDE {self.__class__.__name__} _load_nn_params(): path={path}")
        # log.info(f"[loading parameters from {path}]")
        with open(path, 'r', encoding='utf8') as fp:
            params = json.load(fp)
        if self.debug:
            log.debug(f"INSIDE {self.__class__.__name__} _load_nn_params(): "
                      f"params={params}, GRAPH_PARAMS={self.GRAPH_PARAMS}")

        for p in self.GRAPH_PARAMS:
            if self.__getattribute__(p) != params.get(p) and p not in {'attn', 'attention_mechanism', 'attention_params'}:
                # todo backward-compatible attention serialization
                raise ConfigError(f"`{p}` parameter must be equal to saved"
                                  f" model parameter value `{params.get(p)}`,"
                                  f" but is equal to `{self.__getattribute__(p)}`")

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} _load_nn_params()")

    def save(self, *args, **kwargs) -> None:
        super().save(*args, **kwargs)
        # todo move save_nn_params here?
        self._save_nn_params()

    def _save_nn_params(self) -> None:
        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} _save_nn_params()")

        path = str(self.save_path.with_suffix('.json').resolve())
        if self.debug:
            log.debug(f"INSIDE {self.__class__.__name__} _save_nn_params(): path={path}")
        nn_params = {opt: self.__getattribute__(opt) for opt in self.SERIALIZABLE_FIELDS}
        if self.debug:
            log.debug(f"INSIDE {self.__class__.__name__} _save_nn_params(): nn_params={nn_params}")
        # log.info(f"[saving parameters to {path}]")
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(nn_params, fp)

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} _save_nn_params()")
