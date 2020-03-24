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

from logging import getLogger
from typing import Dict, Any, List, Optional, Union

import numpy as np

import deeppavlov.models.go_bot.templates as templ
from deeppavlov import Chainer
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.models.tf_model import LRScheduledTFModel
from deeppavlov.models.go_bot.data_handler import DataHandler
from deeppavlov.models.go_bot.nn_stuff_handler import NNStuffHandler
from deeppavlov.models.go_bot.tracker import FeaturizedTracker, DialogueStateTracker, MultipleUserStateTracker

log = getLogger(__name__)


@register("go_bot")
class GoalOrientedBot(NNModel):
    """
    The dialogue bot is based on  https://arxiv.org/abs/1702.03274, which
    introduces Hybrid Code Networks that combine an RNN with domain-specific
    knowledge and system action templates.

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

            * **type** – type of attention mechanism, possible values are ``'general'``, ``'bahdanau'``,
              ``'light_general'``, ``'light_bahdanau'``, ``'cs_general'`` and ``'cs_bahdanau'``.
            * **hidden_size** – attention hidden state size.
            * **max_num_tokens** – maximum number of input tokens.
            * **depth** – number of averages used in constrained attentions
              (``'cs_bahdanau'`` or ``'cs_general'``).
            * **action_as_key** – whether to use action from previous timestep as key
              to attention.
            * **intent_as_key** – use utterance intents as attention key or not.
            * **projected_align** – whether to use output projection.
        network_parameters: dictionary with network parameters (for compatibility with release 0.1.1,
            deprecated in the future)

        template_path: file with mapping between actions and text templates
            for response generation.
        template_type: type of used response templates in string format.
        word_vocab: vocabulary of input word tokens
            (:class:`~deeppavlov.core.data.simple_vocab.SimpleVocabulary` recommended).
        bow_embedder: instance of one-hot word encoder
            :class:`~deeppavlov.models.embedders.bow_embedder.BoWEmbedder`.
        embedder: one of embedders from
            :doc:`deeppavlov.models.embedders </apiref/models/embedders>` module.
        slot_filler: component that outputs slot values for a given utterance
            (:class:`~deeppavlov.models.slotfill.slotfill.DstcSlotFillingNetwork`
            recommended).
        intent_classifier: component that outputs intents probability
            distribution for a given utterance (
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
                 tracker: FeaturizedTracker,
                 template_path: str,
                 save_path: str,
                 hidden_size: int = 128,
                 obs_size: int = None,
                 action_size: int = None,
                 dropout_rate: float = 0.,
                 l2_reg_coef: float = 0.,
                 dense_size: int = None,
                 attention_mechanism: dict = None,
                 network_parameters: Optional[Dict[str, Any]] = None,
                 load_path: str = None,
                 template_type: str = "DefaultTemplate",
                 word_vocab: Component = None,
                 bow_embedder: Component = None,
                 embedder: Component = None,
                 slot_filler: Component = None,
                 intent_classifier: Component = None,
                 database: Component = None,
                 api_call_action: str = None,
                 use_action_mask: bool = False,
                 debug: bool = False,
                 **kwargs) -> None:

        self.nn_stuff_handler = NNStuffHandler(load_path=load_path, save_path=save_path, **kwargs)
        self.load_path = self.nn_stuff_handler.load_path
        self.save_path = self.nn_stuff_handler.save_path
        self.data_handler = DataHandler()

        # kwargs here are mostly training hyperparameters

        network_parameters = network_parameters or {}
        if any(p in network_parameters for p in self.DEPRECATED):
            log.warning(f"parameters {self.DEPRECATED} are deprecated,"
                        f" for learning rate schedule documentation see"
                        f" deeppavlov.core.models.lr_scheduled_tf_model"
                        f" or read a github tutorial on super convergence.")
        if 'learning_rate' in network_parameters:
            kwargs['learning_rate'] = network_parameters.pop('learning_rate')
        super().__init__(save_path=None, **kwargs)

        self.tokenizer = tokenizer  # preprocessing

        self.bow_embedder = bow_embedder  # preprocessing?
        self.embedder = embedder  # preprocessing?
        self.word_vocab = word_vocab  # preprocessing?
        self.slot_filler = slot_filler  # another unit of pipeline
        self.intent_classifier = intent_classifier  # another unit of pipeline
        self.use_action_mask = use_action_mask  # feature engineering
        self.debug = debug

        template_path = expand_path(template_path)
        template_type = getattr(templ, template_type)
        log.info(f"[loading templates from {template_path}]")
        self.templates = templ.Templates(template_type).load(template_path)  # upper-level model logic
        self.n_actions = len(self.templates)  # upper-level model logic
        log.info(f"{self.n_actions} templates loaded.")

        self.default_tracker = tracker  # tracker
        self.dialogue_state_tracker = DialogueStateTracker(tracker.slot_names, self.n_actions, hidden_size,
                                                           database)  # tracker

        self.api_call_id = -1  # api call should have smth like action index
        if api_call_action is not None:
            self.api_call_id = self.templates.actions.index(api_call_action)  # upper-level model logic

        self.intents = []
        if isinstance(self.intent_classifier, Chainer):
            self.intents = self.intent_classifier.get_main_component().classes  # upper-level model logic

        new_network_parameters = {
            'hidden_size': hidden_size,
            'action_size': action_size,
            'obs_size': obs_size,
            'dropout_rate': dropout_rate,
            'l2_reg_coef': l2_reg_coef,
            'dense_size': dense_size,
            'attn': attention_mechanism
        }  # network params
        if 'attention_mechanism' in network_parameters:
            network_parameters['attn'] = network_parameters.pop('attention_mechanism')  # network params
        new_network_parameters.update(network_parameters)  # network params
        self._init_network(**new_network_parameters)  # network params

        self.multiple_user_state_tracker = MultipleUserStateTracker()  # tracker
        self.reset()  # tracker

    def _init_network(self,
                      hidden_size: int,
                      action_size: int,
                      obs_size: int,
                      dropout_rate: float,
                      l2_reg_coef: float,
                      dense_size: int,
                      attn: dict) -> None:
        # initialize network
        self.nn_stuff_handler.configure_network_opts(action_size, attn, dense_size, dropout_rate, hidden_size,
                                                     l2_reg_coef,
                                                     obs_size, self.embedder, self.n_actions, self.intent_classifier,
                                                     self.intents,
                                                     self.default_tracker.num_features, self.bow_embedder,
                                                     self.word_vocab)

        # initialize parameters
        self.nn_stuff_handler._configure_network(self)

        if self.nn_stuff_handler.train_checkpoint_exists(self.load_path):
            log.info(f"[initializing `{self.__class__.__name__}` from saved]")
            self.load()
        else:
            log.info(f"[initializing `{self.__class__.__name__}` from scratch]")





    def train_on_batch(self, x: List[dict], y: List[dict]) -> dict:
        b_features, b_emb_context, b_keys, b_u_masks, b_a_masks, b_actions = self.nn_stuff_handler._prepare_data(self, x, y)
        return self.nn_stuff_handler._network_train_on_batch(self, b_features, b_emb_context, b_keys, b_u_masks, b_a_masks, b_actions)

    # todo как инфер понимает из конфига что ему нужно. лёша что-то говорил про дерево
    def _infer(self, tokens: List[str], tracker: DialogueStateTracker) -> List:
        features, emb_context, key = self.data_handler._encode_context(self, tokens, tracker=tracker)
        action_mask = tracker.calc_action_mask(self.api_call_id)
        probs, state_c, state_h = \
            self.nn_stuff_handler._network_call(self, [[features]], [[emb_context]], [[key]],
                              [[action_mask]], [[tracker.network_state[0]]],
                              [[tracker.network_state[1]]],
                              prob=True)  # todo чо за warning кидает ide, почему
        return probs, np.argmax(probs), (state_c, state_h)

    def _infer_dialog(self, contexts: List[dict]) -> List[str]:
        res = []
        self.dialogue_state_tracker.reset_state()
        for context in contexts:
            if context.get('prev_resp_act') is not None:
                previous_act_id = self.data_handler._encode_response(self, context['prev_resp_act'])
                # previous action is teacher-forced
                self.dialogue_state_tracker.update_previous_action(previous_act_id)

            # todo это ответ бд тоже teacher forced?
            self.dialogue_state_tracker.get_ground_truth_db_result_from(context)
            tokens = self.tokenizer([context['text'].lower().strip()])[0]  # todo поч хардкодим ловеркейс

            if callable(self.slot_filler):
                utter_slots = self.slot_filler([tokens])[0]
                self.dialogue_state_tracker.update_state(utter_slots)
            _, predicted_act_id, self.dialogue_state_tracker.network_state = \
                self._infer(tokens, tracker=self.dialogue_state_tracker)

            self.dialogue_state_tracker.update_previous_action(predicted_act_id)
            resp = self.data_handler._decode_response(self, predicted_act_id, self.dialogue_state_tracker)
            res.append(resp)
        return res

    def __call__(self,
                 batch: Union[List[dict], List[str]],
                 user_ids: Optional[List] = None) -> List[str]:
        # batch is a list of utterances
        if isinstance(batch[0], str):
            res = []
            if not user_ids:
                user_ids = ['finn'] * len(batch)
            for user_id, x in zip(user_ids, batch):
                if not self.multiple_user_state_tracker.check_new_user(user_id):
                    self.multiple_user_state_tracker.init_new_tracker(user_id, self.dialogue_state_tracker)

                tracker = self.multiple_user_state_tracker.get_user_tracker(user_id)
                tokens = self.tokenizer([x.lower().strip()])[0]

                if callable(self.slot_filler):
                    utter_slots = self.slot_filler([tokens])[0]
                    tracker.update_state(utter_slots)

                _, predicted_act_id, tracker.network_state = \
                    self._infer(tokens, tracker=tracker)

                tracker.update_previous_action(predicted_act_id)

                # if made api_call, then respond with next prediction
                if predicted_act_id == self.api_call_id:
                    tracker.make_api_call()

                    _, predicted_act_id, tracker.network_state = \
                        self._infer(tokens, tracker=tracker)

                    tracker.update_previous_action(predicted_act_id)

                resp = self.data_handler._decode_response(self, predicted_act_id, tracker)
                res.append(resp)
            return res
        # batch is a list of dialogs, user_ids ignored
        # todo: что значит коммент выше, почему узер идс игноред
        return [self._infer_dialog(x) for x in batch]

    def reset(self, user_id: Union[None, str, int] = None) -> None:
        self.multiple_user_state_tracker.reset(
            user_id)  # todo а чо, у нас всё что можно закешить лежит в мультиюхертрекере?
        if self.debug:
            log.debug("Bot reset.")

    # region helping stuff
    def load(self, *args, **kwargs) -> None:
        self.nn_stuff_handler._load_nn_params(self)
        super().load(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        super().save(*args, **kwargs)
        self.nn_stuff_handler._save_nn_params(self)

    def process_event(self, event_name, data) -> None:
        super().process_event(event_name, data)

    # endregion helping stuff
