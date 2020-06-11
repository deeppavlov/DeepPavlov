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
from typing import Dict, Any

import numpy as np

from deeppavlov.core.models.component import Component
from deeppavlov.models.go_bot.nlg.nlg_manager import NLGManagerInterface
from deeppavlov.models.go_bot.policy.dto.policy_network_params import PolicyNetworkParams
from deeppavlov.models.go_bot.tracker.dto.dst_knowledge import DSTKnowledge
from deeppavlov.models.go_bot.tracker.featurized_tracker import FeaturizedTracker

log = getLogger(__name__)


class DialogueStateTracker(FeaturizedTracker):
    def get_current_knowledge(self) -> DSTKnowledge:
        state_features = self.get_features()
        context_features = self.calc_context_features()
        knowledge = DSTKnowledge(self.prev_action,
                                 state_features, context_features,
                                 self.api_call_id,
                                 self.n_actions)
        return knowledge

    def __init__(self,
                 slot_names, n_actions: int, api_call_id: int, hidden_size: int, database: Component = None) -> None:
        super().__init__(slot_names)
        self.hidden_size = hidden_size
        self.database = database
        self.n_actions = n_actions
        self.api_call_id = api_call_id

        self.reset_state()

    @staticmethod
    def from_gobot_params(parent_tracker: FeaturizedTracker,
                          nlg_manager: NLGManagerInterface,
                          policy_network_params: PolicyNetworkParams,
                          database: Component):
        return DialogueStateTracker(parent_tracker.slot_names,
                                    nlg_manager.num_of_known_actions(), nlg_manager.get_api_call_action_id(),
                                    policy_network_params.hidden_size,
                                    database)

    def reset_state(self):
        super().reset_state()
        self.db_result = None
        self.current_db_result = None
        self.prev_action = np.zeros(self.n_actions, dtype=np.float32)
        self._reset_network_state()

    def _reset_network_state(self):
        self.network_state = (
            np.zeros([1, self.hidden_size], dtype=np.float32),
            np.zeros([1, self.hidden_size], dtype=np.float32)
        )

    def update_previous_action(self, prev_act_id: int) -> None:
        self.prev_action *= 0.
        self.prev_action[prev_act_id] = 1.

    # todo oserikov это стоит переписать
    def update_ground_truth_db_result_from_context(self, context: Dict[str, Any]):
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

    def calc_action_mask(self) -> np.ndarray:
        mask = np.ones(self.n_actions, dtype=np.float32)

        if np.any(self.prev_action):
            prev_act_id = np.argmax(self.prev_action)
            if prev_act_id == self.api_call_id:
                mask[prev_act_id] = 0.

        return mask

    def calc_context_features(self):
        # todo некрасиво
        current_db_result = self.current_db_result
        db_result = self.db_result
        dst_state = self.get_state()

        result_matches_state = 0.
        if current_db_result is not None:
            matching_items = dst_state.items()
            result_matches_state = all(v == db_result.get(s)
                                       for s, v in matching_items
                                       if v != 'dontcare') * 1.
        context_features = np.array([
            bool(current_db_result) * 1.,
            (current_db_result == {}) * 1.,
            (db_result is None) * 1.,
            bool(db_result) * 1.,
            (db_result == {}) * 1.,
            result_matches_state
        ], dtype=np.float32)
        return context_features

    def _update_db_result(self):
        if self.current_db_result is not None:
            self.db_result = self.current_db_result

    def fill_current_state_with_db_results(self) -> dict:
        slots = self.get_state()
        if self.db_result:
            for k, v in self.db_result.items():
                slots[k] = str(v)
        return slots


class MultipleUserStateTrackersPool(object):
    def __init__(self, base_tracker: DialogueStateTracker):
        self._ids_to_trackers = {}
        self.base_tracker = base_tracker

    def check_new_user(self, user_id: int) -> bool:
        return user_id in self._ids_to_trackers

    def get_user_tracker(self, user_id: int) -> DialogueStateTracker:
        if not self.check_new_user(user_id):
            raise RuntimeError(f"The user with {user_id} ID is not being tracked")

        tracker = self._ids_to_trackers[user_id]

        # TODO: understand why setting current_db_result to None is necessary
        tracker.current_db_result = None
        return tracker

    def new_tracker(self):
        tracker = DialogueStateTracker(self.base_tracker.slot_names, self.base_tracker.n_actions,
                                       self.base_tracker.api_call_id, self.base_tracker.hidden_size,
                                       self.base_tracker.database)
        return tracker

    def get_or_init_tracker(self, user_id: int):
        if not self.check_new_user(user_id):
            self.init_new_tracker(user_id, self.base_tracker)

        return self.get_user_tracker(user_id)

    def init_new_tracker(self, user_id: int, tracker_entity: DialogueStateTracker) -> None:
        # TODO: implement a better way to init a tracker
        # todo deprecated. The whole class should follow AbstractFactory or Pool pattern?
        tracker = DialogueStateTracker(
            tracker_entity.slot_names,
            tracker_entity.n_actions,
            tracker_entity.api_call_id,
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
