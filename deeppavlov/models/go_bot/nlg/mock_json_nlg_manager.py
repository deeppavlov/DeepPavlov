import json
from itertools import combinations
from pathlib import Path
from typing import Union, Dict, List, Sequence

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.models.go_bot.nlg.nlg_manager import log
from deeppavlov.models.go_bot.nlg.nlg_manager_interface import NLGManagerInterface

@register("gobot_json_nlg_manager")
class MockJSONNLGManager(NLGManagerInterface):
    def get_api_call_action_id(self):
        return self._api_call_id

    # todo inheritance
    # todo force a2id, id2a mapping to be persistent for same configs

    def __init__(self,
                 actions2slots_path: Union[str, Path],
                 api_call_action: str,
                 data_path: Union[str, Path],
                 debug=False):
        self.debug = debug

        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"actions2slots_path={actions2slots_path}, "
                      f"api_call_action={api_call_action}, debug={debug}")

        individual_actions2slots = self._load_actions2slots_mapping(actions2slots_path)
        possible_actions_combinations_tuples = sorted(
            set(actions_combination_tuple
                for actions_combination_tuple
                in self._extract_actions_combinations(data_path)),
            key = lambda x: '+'.join(x))

        self.action_tuples2ids = {action_tuple: action_tuple_idx
                                  for action_tuple_idx, action_tuple
                                  in enumerate(possible_actions_combinations_tuples)}  # todo: typehint tuples somehow
        self.ids2action_tuples = {v: k for k, v in self.action_tuples2ids.items()}

        self.action_tuples_ids2slots = {}  # todo: typehint tuples somehow
        for actions_combination_tuple in possible_actions_combinations_tuples:
            actions_combination_slots = set(slot
                                            for action in actions_combination_tuple
                                            for slot in individual_actions2slots.get(action, []))
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


    def _extract_actions_combinations(self, dataset_path: Union[str, Path]):
        dataset_path = expand_path(dataset_path)
        dataset = DSTC2DatasetReader.read(data_path=dataset_path, dialogs=True)
        actions_combinations = set()
        for dataset_split in dataset.values():
            for dialogue in dataset_split:
                for user_input, system_response in dialogue:
                    actions_tuple = tuple(system_response["act"].split('+'))
                    actions_combinations.add(actions_tuple)
        return actions_combinations

    @staticmethod
    def _load_actions2slots_mapping(actions2slots_json_path) -> Dict[str, str]:
        actions2slots_json_path = expand_path(actions2slots_json_path)
        with open(actions2slots_json_path, encoding="utf-8") as actions2slots_json_f:
            actions2slots = json.load(actions2slots_json_f)
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

        response_di = {'+'.join(self.ids2action_tuples[actions_tuple_id]):
                           {slot_name: tracker_slotfilled_state.get(slot_name, "unk")
                           for slot_name in slots_to_log}}

        return json.dumps(response_di)

    def num_of_known_actions(self) -> int:
        """
        :returns: the number of actions known to the NLG module
        """
        return len(self.action_tuples2ids.keys())