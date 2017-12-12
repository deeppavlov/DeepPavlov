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

from deeppavlov.common.registry import register_model
from deeppavlov.data.utils import load_vocab
from deeppavlov.models.inferable import Inferable
from deeppavlov.models.trainable import Trainable

from .network import HybridCodeNetworkModel
from .embedder import FasttextUtteranceEmbed
from .bow import BoW_encoder
from .templates import Templates, DualTemplate
from .tracker import DefaultTracker
from .preprocess import SpacyTokenizer

from .metrics import DialogMetrics


@register_model("hcn_new")
class HybridCodeNetworkBot(Inferable, Trainable):

    def __init__(self, vocab_path, template_path, slot_names,
                 template_type:Type=DualTemplate,
                 bow_encoder:Type=BoW_encoder,
                 embedder:Type=FasttextUtteranceEmbed,
                 tokenizer:Type=SpacyTokenizer,
                 tracker:Type=DefaultTracker,
                 network:Type=HybridCodeNetworkModel,
                 use_action_mask=False):

        self.episode_done = True
        self.action_mask = use_action_mask
# TODO: infer slot names from dataset
        self.slot_names = slot_names
        self.bow_encoder = bow_encoder
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.tracker = tracker
        self.network = network

        self.vocab = load_vocab(vocab_path)
        self.templates = Templates(template_type).load(template_path)
        print("[using {} templates from `{}`]"\
              .format(len(self.templates), template_path))

        # initialize slot filler
        self.slot_model = self._load_slot_model('')

        # intialize parameters
        self.db_result = None
        self.n_actions = len(self.templates)
        self.emb_size = self.embedder.dim
        self.prev_action = np.zeros(self.n_actions, dtype=np.float32)

        # initialize metrics
        self.metrics = DialogMetrics(self.n_actions)

        opt = {
            'action_size': self.n_actions,
            'obs_size': 4 + len(self.vocab) + self.emb_size +\
            2 * self.tracker.state_size + self.n_actions
        }
        #self.network = HybridCodeNetworkModel(opt)

    def _load_slot_model(self, slot_model):
# TODO: import slot model
        return None

    def _get_slots(self, text):
        if self.slot_model is not None:
            self.slot_model.observe({
                'text': text,
                'episode_done': True
            })
            return self.slot_model.act()
        return {}

    def _encode_context(self, context, db_result=None):
        # tokenize input
        tokenized = ' '.join(self.tokenizer.infer(context))

        # Bag of words features
        bow_features = self.bow_encoder.infer(tokenized, self.vocab)
        bow_features = bow_features.astype(np.float32)

        # Embeddings
        emb_features = self.embedder.infer(tokenized)

        # Text entity features
        self.tracker.update_state(self._get_slots(tokenized))
        ent_features = self.tracker.infer()

        # Other features
        context_features = np.array([(db_result == {}) * 1.,
                                     (self.db_result == {}) * 1.],
                                    dtype=np.float32)

        return np.hstack((bow_features, emb_features, ent_features,
                          context_features, self.prev_action))[np.newaxis, :]

    def _encode_response(self, response, act):
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
        if self.action_mask:
# TODO: non-ones action mask
            for a_id in range(self.n_actions):
                tmpl = str(self.templates.templates[a_id])
                for entity in re.findall('#{}', tmpl):
                    if entity not in self.tracker.get_state()\
                       and entity not in (self.db_result or {}):
                        action_mask[a_id] = 0
        return action_mask

    def train(self, data, num_epochs, num_tr_dialogs, acc_threshold=0.99):
        print('\n:: training started\n')

        tr_data = data[:num_tr_dialogs]
        eval_data = data[num_tr_dialogs:]

        for j in range(num_epochs):

            self.reset_metrics()

            for dialog in tr_data:

                self.reset()
                self.metrics.n_dialogs += 1

                for turn in dialog:
                    self.db_result = self.db_result or turn.get('db_result')
                    action_id = self._encode_response(turn['response'],
                                                      turn['act'])

                    loss, pred_id = self.network.train(
                        self._encode_context(turn['context'],
                                             turn.get('db_result')),
                        action_id,
                        self._action_mask()
                    )

                    self.prev_action *= 0.
                    self.prev_action[pred_id] = 1.

                    pred = self._decode_response(pred_id).lower()
                    true = self.tokenizer.infer(turn['response'].lower().split())

                    # update metrics
                    self.metrics.n_examples += 1
                    self.metrics.train_loss += loss
                    self.metrics.conf_matrix[pred_id, action_id] += 1
                    self.metrics.n_corr_examples += int(pred == true)
#TODO: update dialog metrics
            print('\n\n:: {}.train {}'.format(j + 1, self.metrics.report()))

            metrics = self.evaluate(eval_data)
            print(':: {}.valid {}'.format(j + 1, metrics.report()))

            if metrics.action_accuracy > acc_threshold:
                print('Accuracy is {}, stopped training.'\
                      .format(metrics.action_accuracy))
                break
        self.save()

    def infer(self, context, db_result=None):
        probs, pred_id = self.network.infer(
            self._encode_context(context, db_result),
            self._action_mask()
        )
        self.prev_action *= 0.
        self.prev_action[pred_id] = 1.
        return self._decode_response(pred_id)

    def evaluate(self, eval_data):
        metrics = DialogMetrics(self.n_actions)

        for dialog in eval_data:

            self.reset()
            metrics.n_dialogs += 1

            for turn in dialog:
                self.db_result = self.db_result or turn.get('db_result')

                probs, pred_id = self.network.infer(
                    self._encode_context(turn['context']),
                    self._action_mask()
                )

                self.prev_action *= 0.
                self.prev_action[pred_id] = 1.

                pred = self._decode_response(pred_id).lower()
                true = self.tokenizer.infer(turn['response'].lower().split())

                # update metrics
                metrics.n_examples += 1
                y = self._encode_response(turn['response'], turn['act'])
                metrics.conf_matrix[pred_id, y] += 1
                metrics.n_corr_examples += int(pred == true)
        return metrics

    def reset(self):
        self.tracker.reset_state()
        self.db_result = None
        self.prev_action *= 0.
        self.network.reset_state()

    def report(self):
        return self.metrics.report()

    def reset_metrics(self):
        self.metrics.reset()

    def save(self):
        """Save the parameters of the agent to a file."""
        self.network.save()

    def __exit__(self, type, value, traceback):
        self.network.__exit__()
