import json
from itertools import combinations
from pathlib import Path
from typing import Union, Dict

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.go_bot.nlg.nlg_manager import log
from deeppavlov.models.go_bot.nlg.nlg_manager_interface import NLGManagerInterface


class JSONMockOutputNLG(NLGManagerInterface):
    # todo inheritance

    def __init__(self, actions2slots_path: Union[str, Path], api_call_action: str, debug=False):
        self.debug = debug

        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"actions2slots_path={actions2slots_path}, "
                      f"api_call_action={api_call_action}, debug={debug}")

        individual_actions2slots = self._load_actions2slots_mapping(actions2slots_path)
        all_possible_actions_combinations_tuples = self._powerset(individual_actions2slots.keys())

        self.action_tuples2ids = {action_tuple: action_tuple_idx
                                  for action_tuple_idx, action_tuple
                                  in enumerate(all_possible_actions_combinations_tuples)}  # todo: typehint tuples somehow

        self.action_tuples_ids2slots = {}  # todo: typehint tuples somehow
        for actions_combination_tuple in all_possible_actions_combinations_tuples:
            actions_combination_slots = set(individual_actions2slots[action]
                                            for action in actions_combination_tuple)
            actions_combination_tuple_id = self.action_tuples2ids[actions_combination_tuple]
            self.action_tuples_ids2slots[actions_combination_tuple_id] = actions_combination_slots

        self._api_call_id = -1
        if api_call_action is not None:
            api_call_action_as_tuple = (api_call_action,)
            self._api_call_id = self.action_tuples2ids[api_call_action_as_tuple]

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} init(): "
                      f"actions2slots_path={actions2slots_path}, "
                      f"api_call_action={api_call_action}, debug={debug}")


    @staticmethod
    def _load_actions2slots_mapping(actions2slots_json_path) -> Dict[str, str]:
        actions2slots_json_path = expand_path(actions2slots_json_path)
        actions2slots = json.load(actions2slots_json_path)
        return actions2slots

    @staticmethod
    def _powerset(iterable):
        all_the_combinations = []
        for powerset_size in range(0, len(iterable) + 1):
            for subset in combinations(iterable, powerset_size):
                all_the_combinations.append(tuple(subset))
        return all_the_combinations

    def get_action_id(self, action_text: str) -> int:
        # todo: docstring

        actions_tuple = tuple(action_text.split('+'))
        return self.action_tuples2ids[actions_tuple]  # todo unhandled exception when not found

    def decode_response(self, actions_tuple_id: int, tracker_slotfilled_state: dict) -> str:
        # todo docstring
        slots_to_log = self.action_tuples_ids2slots[actions_tuple_id]

        response_di = {slot_name: tracker_slotfilled_state.get(slot_name, "unk")
                       for slot_name in slots_to_log}

        return json.dumps(response_di)

    def num_of_known_actions(self) -> int:
        """
        :returns: the number of actions known to the NLG module
        """
        return len(self.action_tuples2ids.keys())