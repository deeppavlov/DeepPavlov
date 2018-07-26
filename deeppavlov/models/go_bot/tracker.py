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

import numpy as np

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register('default_tracker')
class DefaultTracker(Component):

    def __init__(self, slot_names):
        self.slot_names = list(slot_names)
        self.reset_state()

    @property
    def state_size(self):
        return len(self.slot_names)

    @property
    def num_features(self):
        return self.state_size

    def reset_state(self):
        self.history = []
        self.curr_feats = np.zeros(self.num_features, dtype=np.float32)

    def update_state(self, slots):
        def _filter(slots):
            return filter(lambda s: s[0] in self.slot_names, slots)
        if type(slots) == list:
            self.history.extend(_filter(slots))
        elif type(slots) == dict:
            for slot, value in _filter(slots.items()):
                self.history.append((slot, value))
        self.curr_feats = self._binary_features()
        return self

    def get_state(self):
        lasts = {}
        for slot, value in self.history:
            lasts[slot] = value
        return lasts

    def _binary_features(self):
        feats = np.zeros(self.state_size, dtype=np.float32)
        lasts = self.get_state()
        for i, slot in enumerate(self.slot_names):
            if slot in lasts:
                feats[i] = 1.
        return feats

    def __call__(self):
        return self.curr_feats


@register('featurized_tracker')
class FeaturizedTracker(Component):

    def __init__(self, slot_names, *args, **kwargs):
        self.slot_names = list(slot_names)
        self.reset_state()

    @property
    def state_size(self):
        return len(self.slot_names)

    @property
    def num_features(self):
        return self.state_size * 3 + 3

    def reset_state(self):
        self.history = []
        self.curr_feats = np.zeros(self.num_features, dtype=np.float32)

    def update_state(self, slots):
        def _filter(slots):
            return filter(lambda s: s[0] in self.slot_names, slots)
        prev_state = self.get_state()
        if type(slots) == list:
            self.history.extend(_filter(slots))
        elif type(slots) == dict:
            for slot, value in _filter(slots.items()):
                self.history.append((slot, value))
        bin_feats = self._binary_features()
        diff_feats = self._diff_features(prev_state)
        new_feats = self._new_features(prev_state)
        self.curr_feats = np.hstack((bin_feats,
                                     diff_feats,
                                     new_feats,
                                     np.sum(bin_feats),
                                     np.sum(diff_feats),
                                     np.sum(new_feats)))
        return self

    def get_state(self):
        lasts = {}
        for slot, value in self.history:
            lasts[slot] = value
        return lasts

    def _binary_features(self):
        feats = np.zeros(self.state_size, dtype=np.float32)
        lasts = self.get_state()
        for i, slot in enumerate(self.slot_names):
            if slot in lasts:
                feats[i] = 1.
        return feats

    def _diff_features(self, state):
        feats = np.zeros(self.state_size, dtype=np.float32)
        curr_state = self.get_state()
        for i, slot in enumerate(self.slot_names):
            if (slot in curr_state) and (slot in state) and\
                    (curr_state[slot] != state[slot]):
                feats[i] = 1.
        return feats

    def _new_features(self, state):
        feats = np.zeros(self.state_size, dtype=np.float32)
        curr_state = self.get_state()
        for i, slot in enumerate(self.slot_names):
            if (slot in curr_state) and (slot not in state):
                feats[i] = 1.
        return feats

    def __call__(self):
        return self.curr_feats
