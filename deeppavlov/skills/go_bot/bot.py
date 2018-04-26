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

import re

import numpy as np
from typing import Type

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.models.tokenizers.spacy_tokenizer import StreamSpacyTokenizer
from deeppavlov.models.trackers.default_tracker import DefaultTracker
from deeppavlov.skills.go_bot.network import GoalOrientedBotNetwork
from deeppavlov.skills.go_bot.templates import Templates, DualTemplate
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("go_bot")
class GoalOrientedBot(NNModel):
    def __init__(self,
                 template_path,
                 network_parameters,
                 template_type: Type = DualTemplate,
                 tokenizer: Type = StreamSpacyTokenizer,
                 tracker: Type = DefaultTracker,
                 bow_embedder=None,
                 embedder=None,
                 slot_filler=None,
                 intent_classifier=None,
                 use_action_mask=False,
                 db_result_during_interaction=None,
                 debug=False,
                 save_path=None,
                 word_vocab=None,
                 vocabs=None,
                 **kwargs):

        super().__init__(save_path=save_path, mode=kwargs['mode'])

        self.episode_done = True
        self.use_action_mask = use_action_mask
        self.debug = debug
        self.slot_filler = slot_filler
        self.intent_classifier = intent_classifier
        self.bow_embedder = bow_embedder
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.tracker = tracker
        self.word_vocab = word_vocab or vocabs['word_vocab']
        self.interact_db_result = db_result_during_interaction

        template_path = expand_path(template_path)
        log.info("[loading templates from {}]".format(template_path))
        self.templates = Templates(template_type).load(template_path)
        self.n_actions = len(self.templates)
        log.info("{} templates loaded".format(self.n_actions))

        self.network = self._init_network(network_parameters)

        self.reset()

    def _init_network(self, params):
        # initialize network
        obs_size = 4 + self.tracker.num_features + self.n_actions
        if callable(self.bow_embedder):
            obs_size += len(self.word_vocab)
        if callable(self.embedder):
            obs_size += self.embedder.dim
        if callable(self.intent_classifier):
            obs_size += self.intent_classifier.n_classes
        log.info("Calculated input size for `GoalOrientedBotNetwork` is {}"
                 .format(obs_size))
        if 'obs_size' not in params:
            params['obs_size'] = obs_size
        if 'action_size' not in params:
            params['action_size'] = self.n_actions

        attn = params.get('attention_mechanism')
        if attn:
            attn['token_size'] = attn.get('token_size') or self.embedder.dim
            attn['action_as_key'] = attn.get('action_as_key', False)
            attn['intent_as_key'] = attn.get('intent_as_key', False)

            key_size = 0
            if attn['action_as_key']:
                key_size += self.n_actions
            if attn['intent_as_key'] and callable(self.intent_classifier):
                key_size += self.intent_classifier.n_classes
            key_size = key_size or 1
            attn['key_size'] = attn.get('key_size') or key_size

            params['attention_mechanism'] = attn
        return GoalOrientedBotNetwork(**params)

    def _encode_context(self, context, db_result=None):
        # tokenize input
        tokens = self.tokenizer([context.lower().strip()])[0]
        tokenized = ' '.join(tokens)
        if self.debug:
            log.debug("Text tokens = `{}`".format(tokens))

        # Bag of words features
        bow_features = []
        if callable(self.bow_embedder):
            bow_features = self.bow_embedder([tokens], self.word_vocab)[0]
            bow_features = bow_features.astype(np.float32)

        # Embeddings
        emb_features = []
        emb_context = np.array([], dtype=np.float32)
        if callable(self.embedder):
            if self.network.attn:
                if tokens:
                    pad = np.zeros((self.network.attn.max_num_tokens,
                                    self.network.attn.token_size),
                                   dtype=np.float32)
                    sen = np.array(self.embedder([tokens])[0])
                    # TODO : Unsupport of batch_size more than 1
                    emb_context = np.concatenate((pad, sen))
                    emb_context = emb_context[-self.network.attn.max_num_tokens:]
                else:
                    emb_context = \
                        np.zeros((self.network.attn.max_num_tokens,
                                  self.network.attn.token_size),
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
            intent_features = \
                self.intent_classifier([tokenized], predict_proba=True).ravel()
            if self.debug:
                log.debug("Predicted intent = `{}`"
                          .format(self.intent_classifier([tokenized])))

        # Text entity features
        if callable(self.slot_filler):
            self.tracker.update_state(self.slot_filler([tokenized])[0])
            if self.debug:
                log.debug("Slot vals: {}".format(str(self.slot_filler(tokenized))))

        state_features = self.tracker()

        # Other features
        new_db_result = db_result if db_result is not None else self.db_result
        context_features = np.array([bool(self.db_result) * 1.,
                                     bool(new_db_result) * 1.,
                                     (self.db_result == {}) * 1.,
                                     (new_db_result == {}) * 1.],
                                    dtype=np.float32)

        if self.debug:
            debug_msg = "num bow features = {}, ".format(len(bow_features)) +\
                        "num emb features = {}, ".format(len(emb_features)) +\
                        "num intent features = {}, ".format(len(intent_features)) +\
                        "num state features = {}, ".format(len(state_features)) +\
                        "num context features = {}, ".format(len(context_features)) +\
                        "prev_action shape = {}".format(len(self.prev_action))
            log.debug(debug_msg)

        attn_key = np.array([], dtype=np.float32)
        if self.network.attn:
            if self.network.attn.action_as_key:
                attn_key = np.hstack((attn_key, self.prev_action))
            if self.network.attn.intent_as_key:
                attn_key = np.hstack((attn_key, intent_features))
            if len(attn_key) == 0:
                attn_key = np.array([1], dtype=np.float32)

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

        return template.generate_text(slots)

    def _action_mask(self):
        action_mask = np.ones(self.n_actions, dtype=np.float32)
        if self.use_action_mask:
            known_entities = {**self.tracker.get_state(), **(self.db_result or {})}
            for a_id in range(self.n_actions):
                tmpl = str(self.templates.templates[a_id])
                for entity in set(re.findall('#([A-Za-z]+)', tmpl)):
                    if entity not in known_entities:
                        action_mask[a_id] = 0
        return action_mask

    def train_on_batch(self, x, y):
        b_features, b_u_masks, b_a_masks, b_actions = [], [], [], []
        b_emb_context, b_keys = [], []  # for attention
        max_num_utter = max(len(d_contexts) for d_contexts in x)
        for d_contexts, d_responses in zip(x, y):
            self.reset()
            d_features, d_a_masks, d_actions = [], [], []
            d_emb_context, d_key = [], []  # for attention
            for context, response in zip(d_contexts, d_responses):
                features, emb_context, key = \
                    self._encode_context(context['text'], context.get('db_result'))
                if context.get('db_result') is not None:
                    self.db_result = context['db_result']
                d_features.append(features)
                d_emb_context.append(emb_context)
                d_key.append(key)

                action_id = self._encode_response(response['act'])
                # previous action is teacher-forced here
                self.prev_action *= 0.
                self.prev_action[action_id] = 1.
                d_actions.append(action_id)

                d_a_masks.append(self._action_mask())

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

        self.network.train_on_batch(b_features, b_emb_context, b_keys, b_u_masks, 
                                    b_a_masks, b_actions)

    def _infer(self, context, db_result=None, prob=True):
        features, emb_context, key = self._encode_context(context, db_result)
        probs = self.network(
            [[features]], [[emb_context]], [[key]], [[self._action_mask()]], prob=True
        )
        pred_id = np.argmax(probs)
        if db_result is not None:
            self.db_result = db_result

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

            res.append(self._infer(context['text'], context.get('db_result')))
        return res

    def __call__(self, batch):
        if isinstance(batch[0], str):
            if self.tracker.get_state():
                self.db_result = self.interact_db_result
            return [self._infer(x) for x in batch]
        return [self._infer_dialog(x) for x in batch]

    def reset(self):
        self.tracker.reset_state()
        self.db_result = None
        self.prev_action = np.zeros(self.n_actions, dtype=np.float32)
        self.network.reset_state()

    def save(self):
        """Save the parameters of the model to a file."""
        self.network.save()

    def shutdown(self):
        self.network.shutdown()
        self.slot_filler.shutdown()

    def load(self):
        pass
