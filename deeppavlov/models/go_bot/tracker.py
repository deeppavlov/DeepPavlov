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

from abc import ABCMeta, abstractmethod
from logging import getLogger
from typing import List, Dict, Union, Tuple, Any, Iterator

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


class Tracker(metaclass=ABCMeta):
    """
    An abstract class for trackers: a model that holds a dialogue state and
    generates state features.
    """

    @abstractmethod
    def update_state(self, slots: Union[List[Tuple[str, Any]], Dict[str, Any]]) -> None:
        """
        Updates dialogue state with new ``slots``, calculates features.

        Returns:
            Tracker: ."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: dictionary with current slots and their values."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Resets dialogue state"""
        pass

    @abstractmethod
    def get_features(self) -> np.ndarray:
        """
        Returns:
            np.ndarray[float]: numpy array with calculates state features."""
        pass


@register('featurized_tracker')
class FeaturizedTracker(Tracker):
    """
    Tracker that overwrites slots with new values.
    Features are binary features (slot is present/absent) plus difference features
    (slot value is (the same)/(not the same) as before last update) and count
    features (sum of present slots and sum of changed during last update slots).

    Parameters:
        slot_names: list of slots that should be tracked.
    """

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

    def update_state(self, slots):
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
        lasts = {}
        for slot, value in self.history:
            lasts[slot] = value
        return lasts

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


class DialogueStateTracker(FeaturizedTracker):
    def __init__(self, slot_names, n_actions: int, hidden_size: int, database: Component = None) -> None:
        super().__init__(slot_names)
        self.db_result = None
        self.current_db_result = None
        self.database = database

        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.prev_action = np.zeros(n_actions, dtype=np.float32)

        self.network_state = (
            np.zeros([1, hidden_size], dtype=np.float32),
            np.zeros([1, hidden_size], dtype=np.float32)
        )

    def reset_state(self):
        super().reset_state()
        self.db_result = None
        self.current_db_result = None
        self.prev_action = np.zeros(self.n_actions, dtype=np.float32)

        self.network_state = (
            np.zeros([1, self.hidden_size], dtype=np.float32),
            np.zeros([1, self.hidden_size], dtype=np.float32)
        )

    def update_previous_action(self, prev_act_id: int) -> None:
        self.prev_action *= 0.
        self.prev_action[prev_act_id] = 1.

    def get_ground_truth_db_result_from(self, context: Dict[str, Any]):
        self.current_db_result = context.get('db_result', None)
        self._update_db_result()

    def make_api_call(self) -> None:
        slots = self.get_state()
        db_results = []
        if self.database is not None:

            # filter slot keys with value equal to 'dontcare' as
            # there is no such value in database records
            # and remove unknown slot keys (for example, 'this' in dstc2 tracker)
            db_slots = {
                s: v for s, v in slots.items() if v != 'dontcare' and s in self.database.keys
            }

            db_results = self.database([db_slots])[0]

            # filter api results if there are more than one
            # TODO: add sufficient criteria for database results ranking
            if len(db_results) > 1:
                db_results = [r for r in db_results if r != self.db_result]
        else:
            log.warning("No database specified.")

        log.info(f"Made api_call with {slots}, got {len(db_results)} results.")
        self.current_db_result = {} if not db_results else db_results[0]
        self._update_db_result()

    def calc_action_mask(self, api_call_id: int) -> np.ndarray:
        mask = np.ones(self.n_actions, dtype=np.float32)

        if np.any(self.prev_action):
            prev_act_id = np.argmax(self.prev_action)
            if prev_act_id == api_call_id:
                mask[prev_act_id] = 0.

        return mask

    def _update_db_result(self):
        if self.current_db_result is not None:
            self.db_result = self.current_db_result


class MultipleUserStateTracker(object):
    def __init__(self):
        self._ids_to_trackers = {}

    def check_new_user(self, user_id: int) -> bool:
        return user_id in self._ids_to_trackers

    def get_user_tracker(self, user_id: int) -> DialogueStateTracker:
        if not self.check_new_user(user_id):
            raise RuntimeError(f"The user with {user_id} ID is not being tracked")

        tracker = self._ids_to_trackers[user_id]

        # TODO: understand why setting current_db_result to None is necessary
        tracker.current_db_result = None
        return tracker

    def init_new_tracker(self, user_id: int, tracker_entity: DialogueStateTracker) -> None:
        # TODO: implement a better way to init a tracker
        tracker = DialogueStateTracker(
            tracker_entity.slot_names,
            tracker_entity.n_actions,
            tracker_entity.hidden_size,
            tracker_entity.database
        )

        self._ids_to_trackers[user_id] = tracker

    def reset(self, user_id: int = None) -> None:
        if user_id is not None and not self.check_new_user(user_id):
            raise RuntimeError(f"The user with {user_id} ID is not being tracked")

        if user_id is not None:
            self._ids_to_trackers[user_id].reset_state()
        else:
            self._ids_to_trackers.clear()

