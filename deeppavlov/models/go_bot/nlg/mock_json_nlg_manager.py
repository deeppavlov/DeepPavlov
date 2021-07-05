import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Union, Dict, List, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register, get_model
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.dataset_readers.dto.rasa.domain_knowledge import DomainKnowledge
from deeppavlov.models.go_bot.dto.dataset_features import BatchDialoguesFeatures
from deeppavlov.models.go_bot.nlg.dto.json_nlg_response import JSONNLGResponse, VerboseJSONNLGResponse
from deeppavlov.models.go_bot.nlg.nlg_manager import log
from deeppavlov.models.go_bot.nlg.nlg_manager_interface import NLGManagerInterface
from deeppavlov.models.go_bot.policy.dto.policy_prediction import PolicyPrediction
import random

@register("gobot_json_nlg_manager")
class MockJSONNLGManager(NLGManagerInterface):

    # todo inheritance
    # todo force a2id, id2a mapping to be persistent for same configs

    def __init__(self,
                 actions2slots_path: Union[str, Path],
                 api_call_action: str,
                 data_path: Union[str, Path],
                 dataset_reader_class="dstc2_reader",
                 debug=False):
        self.debug = debug

        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"actions2slots_path={actions2slots_path}, "
                      f"api_call_action={api_call_action}, debug={debug}")

        self._dataset_reader = get_model(dataset_reader_class)

        individual_actions2slots = self._load_actions2slots_mapping(actions2slots_path)
        split2domain_i = self._get_domain_info(data_path)
        possible_actions_combinations_tuples = sorted(
            set(actions_combination_tuple
                for actions_combination_tuple
                in self._extract_actions_combinations(split2domain_i)),
            key=lambda x: '+'.join(x))

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

        self.action2slots2text, self.action2slots2values2text =\
            self._extract_templates(split2domain_i)

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} init(): "
                      f"actions2slots_path={actions2slots_path}, "
                      f"api_call_action={api_call_action}, debug={debug}")

    def get_api_call_action_id(self) -> int:
        """
        Returns:
            an ID corresponding to the api call action
        """
        return self._api_call_id

    def _get_domain_info(self, dataset_path: Union[str, Path]):
        dataset_path = expand_path(dataset_path)
        try:
            dataset = self._dataset_reader.read(data_path=dataset_path)
        except:
            dataset = self._dataset_reader.read(data_path=dataset_path,
                                                fmt="yml")
        split2domain = dict()
        for dataset_split, dataset_split_info in dataset.items():
            domain_i: DomainKnowledge = dataset_split_info["domain"]
            split2domain[dataset_split] = domain_i
        return split2domain

    def _extract_actions_combinations(self, split2domain: Dict[str, DomainKnowledge]):
        actions_combinations = set()
        for dataset_split, domain_i in split2domain.items():
            actions_combinations.update({(ac,) for ac in domain_i.known_actions})
        return actions_combinations

    def _extract_templates(self, split2domain: Dict[str, DomainKnowledge]):
        slots_pattern = r'\[(?P<value>\w+)\]\((?P<name>\w+)\)'
        action2slots2text = defaultdict(lambda: defaultdict(list))
        action2slots2values2text = defaultdict(lambda: defaultdict(list))
        for dataset_split, domain_i in split2domain.items():
            actions2texts = domain_i.response_templates
            for action, texts in actions2texts.items():
                action_tuple = (action,)
                texts = [text for text in texts if text]
                for text in texts:
                    used_slots, slotvalue_tuples = set(), set()
                    if isinstance(text, dict):
                        text = text["text"]
                    used_slots_di = dict()
                    for found in re.finditer(slots_pattern, text):
                        used_slots_di = found.groupdict()
                        if not ("name" in used_slots_di.keys() and "value" in used_slots_di.keys()):
                            continue
                        used_slots.update(used_slots_di["name"])
                        slotvalue_tuples.update({used_slots_di["name"]:
                                                 used_slots_di["value"]})

                    used_slots = tuple(sorted(used_slots))
                    slotvalue_tuples = tuple(sorted(slotvalue_tuples))
                    templated_text = re.sub(slots_pattern, '##\g<name>', text)
                    action2slots2text[action_tuple][used_slots].append(templated_text)
                    action2slots2values2text[action_tuple][slotvalue_tuples].append(templated_text)

        return action2slots2text, action2slots2values2text

    def generate_template(self, response_info: VerboseJSONNLGResponse, mode="slots"):
        if mode == "slots":
            response_text = None
            action_tuple = response_info.actions_tuple
            slots = tuple(sorted(response_info.slot_values.keys()))
            response_text = self.action2slots2text.get(action_tuple, {}).get(slots, None)
        else:
            action_tuple = response_info.actions_tuple
            slotvalue_tuples = tuple(sorted(response_info.slot_values.items()))
            response_text = self.action2slots2text.get(action_tuple, {}).get(slotvalue_tuples, None)
        if isinstance(response_text, list):
            response_text = random.choice(response_text)
        for slot_name in response_info.slot_values:
            response_text = response_text.replace(f"##{slot_name}",
                                                  response_info.slot_values[
                                                      slot_name])
        return response_text

    @staticmethod
    def _load_actions2slots_mapping(actions2slots_json_path) -> Dict[str, str]:
        actions2slots_json_path = expand_path(actions2slots_json_path)
        if actions2slots_json_path.exists():
            with open(actions2slots_json_path, encoding="utf-8") as actions2slots_json_f:
                actions2slots = json.load(actions2slots_json_f)
        else:
            actions2slots = dict()
            log.info(f"INSIDE {__class__.__name__} _load_actions2slots_mapping(): "
                      f"actions2slots_json_path={actions2slots_json_path} DOES NOT EXIST. "
                      f"initialized actions2slots mapping with an empty one: {str(actions2slots)}")
        return actions2slots

    def get_action_id(self, action_text: Union[str, Tuple[str, ...]]) -> int:
        """
        Looks up for an ID corresponding to the passed action text.

        Args:
            action_text: the text for which an ID needs to be returned.
        Returns:
            an ID corresponding to the passed action text
        """
        if isinstance(action_text, str):
            actions_tuple = tuple(action_text.split('+'))
        else:
            actions_tuple = action_text
        return self.action_tuples2ids[actions_tuple]  # todo unhandled exception when not found

    def decode_response(self,
                        utterance_batch_features: BatchDialoguesFeatures,
                        policy_prediction: PolicyPrediction,
                        tracker_slotfilled_state: dict) -> JSONNLGResponse:
        """
        Converts the go-bot inference objects to the single output object.

        Args:
            utterance_batch_features: utterance features extracted in go-bot that
            policy_prediction: policy model prediction (predicted action)
            tracker_slotfilled_state: tracker knowledge before the NLG is performed

        Returns:
            The NLG output unit that stores slot values and predicted actions info.
        """
        slots_to_log = self.action_tuples_ids2slots[policy_prediction.predicted_action_ix]

        slots_values = {slot_name: tracker_slotfilled_state.get(slot_name, "unk") for slot_name in slots_to_log}
        actions_tuple = self.ids2action_tuples[policy_prediction.predicted_action_ix]

        response = JSONNLGResponse(slots_values, actions_tuple)
        verbose_response = VerboseJSONNLGResponse.from_json_nlg_response(response)
        verbose_response.policy_prediction = policy_prediction
        response_text = self.generate_template(verbose_response)
        verbose_response.text = response_text
        if utterance_batch_features:
            verbose_response._nlu_responses = utterance_batch_features._nlu_responses
            return verbose_response
        # TripPy Case - Use same return type as nlg_manager, i.e. str
        else:
            return verbose_response.text

    def num_of_known_actions(self) -> int:
        """
        Returns:
            the number of actions known to the NLG module
        """
        return len(self.action_tuples2ids.keys())

    def known_actions(self) -> List:
        """
        Returns:
             the list of actions known to the NLG module
        """
        return list(self.action_tuples2ids.keys())
