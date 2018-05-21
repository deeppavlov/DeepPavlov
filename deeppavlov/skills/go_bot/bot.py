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
from deeppavlov.skills.go_bot.network import GoalOrientedBotNetwork
import deeppavlov.skills.go_bot.templates as templ
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("go_bot")
class GoalOrientedBot(NNModel):
    def __init__(self,
                 template_path,
                 network_parameters,
                 tokenizer,
                 tracker,
                 template_type: str = "BaseTemplate",
                 database=None,
                 api_call_action=None,
                 bow_embedder=None,
                 embedder=None,
                 slot_filler=None,
                 intent_classifier=None,
                 use_action_mask=False,
                 debug=False,
                 save_path=None,
                 word_vocab=None,
                 vocabs=None,
                 **kwargs):

        super().__init__(save_path=save_path, mode=kwargs['mode'])

        self.tokenizer = tokenizer
        self.tracker = tracker
        self.bow_embedder = bow_embedder
        self.embedder = embedder
        self.slot_filler = slot_filler
        self.intent_classifier = intent_classifier
        self.use_action_mask = use_action_mask
        self.debug = debug
        self.word_vocab = word_vocab or vocabs['word_vocab']

        template_path = expand_path(template_path)
        template_type = getattr(templ, template_type)
        log.info("[loading templates from {}]".format(template_path))
        self.templates = templ.Templates(template_type).load(template_path)
        self.n_actions = len(self.templates)
        log.info("{} templates loaded".format(self.n_actions))

        self.database = database
        self.api_call_id = self.templates.actions.index(api_call_action)

        self.intents = []
        if callable(self.intent_classifier):
            # intent_classifier returns y_labels, y_probs
            self.intents = list(self.intent_classifier(["hi"])[1][0].keys())

        self.network = self._init_network(network_parameters)

        self.reset()

    def _init_network(self, params):
        # initialize network
        obs_size = 6 + self.tracker.num_features + self.n_actions
        if callable(self.bow_embedder):
            obs_size += len(self.word_vocab)
        if callable(self.embedder):
            obs_size += self.embedder.dim
        if callable(self.intent_classifier):
            obs_size += len(self.intents)
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
                key_size += len(self.intents)
            key_size = key_size or 1
            attn['key_size'] = attn.get('key_size') or key_size

            params['attention_mechanism'] = attn
        return GoalOrientedBotNetwork(**params)

    def _encode_context(self, context, db_result=None):
        # tokenize input
        tokens = self.tokenizer([context.lower().strip()])[0]
        if self.debug:
            log.debug("Tokenized text= `{}`".format(' '.join(tokens)))

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
            intent, intent_probs = self.intent_classifier([tokens])
            intent_features = np.array([intent_probs[0][i] for i in self.intents],
                                       dtype=np.float32)
            if self.debug:
                log.debug("Predicted intent = `{}`".format(intent[0]))

        attn_key = np.array([], dtype=np.float32)
        if self.network.attn:
            if self.network.attn.action_as_key:
                attn_key = np.hstack((attn_key, self.prev_action))
            if self.network.attn.intent_as_key:
                attn_key = np.hstack((attn_key, intent_features))
            if len(attn_key) == 0:
                attn_key = np.array([1], dtype=np.float32)

        # Text entity features
        if callable(self.slot_filler):
            self.tracker.update_state(self.slot_filler([tokens])[0])
            if self.debug:
                log.debug("Slot vals: {}".format(self.slot_filler([tokens])))

        state_features = self.tracker()

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

    def _action_mask(self, previous_action):
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

    def train_on_batch(self, x, y):
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
                d_a_masks.append(self._action_mask(self.prev_action))

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

        self.network.train_on_batch(b_features, b_emb_context, b_keys, b_u_masks,
                                    b_a_masks, b_actions)

    def _infer(self, context, db_result=None, prob=False):
        if db_result is not None:
            self.db_result = db_result
        features, emb_context, key = self._encode_context(context, db_result)
        action_mask = self._action_mask(self.prev_action)
        probs = self.network(
            [[features]], [[emb_context]], [[key]], [[action_mask]], prob=True
        )
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

            res.append(self._infer(context['text'], context.get('db_result')))
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
                if np.argmax(self.prev_action) == self.api_call_id:
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
        self.network.reset_state()
        if self.debug:
            log.debug("Bot reset.")

    def process_event(self, *args, **kwargs):
        self.network.process_event(*args, **kwargs)

    def save(self):
        """Save the parameters of the model to a file."""
        self.network.save()

    def shutdown(self):
        self.network.shutdown()
        self.slot_filler.shutdown()

    def load(self):
        pass
