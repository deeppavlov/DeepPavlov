from typing import List, Iterator

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.models.go_bot.nlu.dto.nlu_response import NLUResponse
from deeppavlov.models.go_bot.tracker.dto.tracker_knowledge_interface import TrackerKnowledgeInterface
from deeppavlov.models.go_bot.tracker.tracker_interface import TrackerInterface


@register('featurized_tracker')
class FeaturizedTracker(TrackerInterface):
    """
    Tracker that overwrites slots with new values.
    Features are binary features (slot is present/absent) plus difference features
    (slot value is (the same)/(not the same) as before last update) and count
    features (sum of present slots and sum of changed during last update slots).

    Parameters:
        slot_names: list of slots that should be tracked.
    """

    def get_current_knowledge(self) -> TrackerKnowledgeInterface:
        raise NotImplementedError("Featurized tracker lacks get_current_knowledge() method. "
                                  "To be improved in future versions.")

    def __init__(self, slot_names: List[str]) -> None:
        self.slot_names = list(slot_names)
        self.history = []
        self.current_features = None

    @property
    def state_size(self) -> int:
        return len(self.slot_names)

    @property
    def num_features(self) -> int:
        return self.state_size * 3 + 3

    def update_state(self, nlu_response: NLUResponse):
        slots = nlu_response.slots

        if isinstance(slots, list):
            self.history.extend(self._filter(slots))

        elif isinstance(slots, dict):
            for slot, value in self._filter(slots.items()):
                self.history.append((slot, value))

        prev_state = self.get_state()
        bin_feats = self._binary_features()
        diff_feats = self._diff_features(prev_state)
        new_feats = self._new_features(prev_state)

        self.current_features = np.hstack((
            bin_feats,
            diff_feats,
            new_feats,
            np.sum(bin_feats),
            np.sum(diff_feats),
            np.sum(new_feats))
        )

    def get_state(self):
        # lasts = {}
        # for slot, value in self.history:
        #     lasts[slot] = value
        # return lasts
        return dict(self.history)

    def reset_state(self):
        self.history = []
        self.current_features = np.zeros(self.num_features, dtype=np.float32)

    def get_features(self):
        return self.current_features

    def _filter(self, slots) -> Iterator:
        return filter(lambda s: s[0] in self.slot_names, slots)

    def _binary_features(self) -> np.ndarray:
        feats = np.zeros(self.state_size, dtype=np.float32)
        lasts = self.get_state()
        for i, slot in enumerate(self.slot_names):
            if slot in lasts:
                feats[i] = 1.
        return feats

    def _diff_features(self, state) -> np.ndarray:
        feats = np.zeros(self.state_size, dtype=np.float32)
        curr_state = self.get_state()

        for i, slot in enumerate(self.slot_names):
            if slot in curr_state and slot in state and curr_state[slot] != state[slot]:
                feats[i] = 1.

        return feats

    def _new_features(self, state) -> np.ndarray:
        feats = np.zeros(self.state_size, dtype=np.float32)
        curr_state = self.get_state()

        for i, slot in enumerate(self.slot_names):
            if slot in curr_state and slot not in state:
                feats[i] = 1.

        return feats
