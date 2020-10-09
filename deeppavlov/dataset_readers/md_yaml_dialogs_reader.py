# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwaredata
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import re
import tempfile
from collections import defaultdict
from logging import getLogger
from overrides import overrides
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional

from deeppavlov.core.common.file import read_yaml
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader

SLOT2VALUE_PAIRS_TUPLE = Tuple[Tuple[str, Any], ...]

log = getLogger(__name__)


class DomainKnowledge:
    """the DTO-like class to store the domain knowledge from the domain yaml config."""

    known_entities: List
    known_intents: List
    known_actions: List
    known_slots: Dict
    response_templates: Dict
    session_config: Dict

    def __init__(self, domain_knowledge_di: Dict):
        self.known_entities = domain_knowledge_di.get("entities", [])
        self.known_intents = domain_knowledge_di.get("intents", [])
        self.known_actions = domain_knowledge_di.get("actions", [])
        self.known_slots = domain_knowledge_di.get("slots", {})
        self.response_templates = domain_knowledge_di.get("responses", {})
        self.session_config = domain_knowledge_di.get("session_config", {})


@register('md_yaml_dialogs_reader')
class MD_YAML_DialogsDatasetReader(DatasetReader):
    """
    Reads dialogs from dataset composed of ``stories.md``, ``nlu.md``, ``domain.yml`` .

    ``stories.md`` is to provide the dialogues dataset for model to train on. The dialogues
    are represented as user messages labels and system response messages labels: (not texts, just action labels).
    This is so to distinguish the NLU-NLG tasks from the actual dialogues storytelling experience: one
    should be able to describe just the scripts of dialogues to the system.

    ``nlu.md`` is contrariwise to provide the NLU training set irrespective of the dialogues scripts.

    ``domain.yml`` is to desribe the task-specific domain and serves two purposes:
    provide the NLG templates and provide some specific configuration of the NLU
    """

    _USER_SPEAKER_ID = 1
    _SYSTEM_SPEAKER_ID = 2

    VALID_DATATYPES = ('trn', 'val', 'tst')

    @classmethod
    def _data_fname(cls, datatype: str) -> str:
        assert datatype in cls.VALID_DATATYPES, f"wrong datatype name: {datatype}"
        return f"stories-{datatype}.md"

    @classmethod
    @overrides
    def read(cls, data_path: str, dialogs: bool = False) -> Dict[str, List]:
        """
        Parameters:
            data_path: path to read dataset from
            dialogs: flag which indicates whether to output list of turns or
                list of dialogs

        Returns:
            dictionary that contains
            ``'train'`` field with dialogs from ``'stories-trn.md'``,
            ``'valid'`` field with dialogs from ``'stories-val.md'`` and
            ``'test'`` field with dialogs from ``'stories-tst.md'``.
            Each field is a list of tuples ``(x_i, y_i)``.
        """
        domain_fname = "domain.yml"
        nlu_fname = "nlu.md"
        stories_fnames = tuple(cls._data_fname(dt) for dt in cls.VALID_DATATYPES)
        required_fnames = stories_fnames + (nlu_fname, domain_fname)
        for required_fname in required_fnames:
            required_path = Path(data_path, required_fname)
            if not required_path.exists():
                log.error(f"INSIDE MLU_MD_DialogsDatasetReader.read(): "
                          f"{required_fname} not found with path {required_path}")

        domain_knowledge = DomainKnowledge(read_yaml(Path(data_path, domain_fname)))
        intent2slots2text, slot_name2text2value = cls._read_intent2text_mapping(Path(data_path, nlu_fname),
                                                                                domain_knowledge)

        short2long_subsample_name = {"trn": "train",
                                     "val": "valid",
                                     "tst": "test"}

        data = {short2long_subsample_name[subsample_name_short]:
                    cls._read_story(Path(data_path, cls._data_fname(subsample_name_short)),
                                    dialogs, domain_knowledge, intent2slots2text, slot_name2text2value)
                for subsample_name_short in cls.VALID_DATATYPES}

        return data

    @classmethod
    def _read_intent2text_mapping(cls, nlu_fpath: Path, domain_knowledge: DomainKnowledge) \
            -> Tuple[Dict[str, Dict[SLOT2VALUE_PAIRS_TUPLE, List]],
                     Dict[str, Dict[str, str]]]:

        slots_markup_pattern = r"\[" + \
                               r"(?P<slot_value>.*?)" + \
                               r"\]" + \
                               r"\(" + \
                               r"(?P<slot_name>.*?)" + \
                               r"\)"

        intent2slots2text = defaultdict(lambda: defaultdict(list))
        slot_name2text2value = defaultdict(dict)

        curr_intent_name = None

        with open(nlu_fpath) as nlu_f:
            for line in nlu_f:
                if line.startswith("##"):
                    # lines starting with ## are starting section describing new intent type
                    curr_intent_name = line.strip("##").strip().split("intent:", 1)[-1]

                if line.strip().startswith('-'):
                    # lines starting with - are listing the examples of intent texts of the current intent type
                    intent_text_w_markup = line.strip().strip('-').strip()
                    line_slots_found = re.finditer(slots_markup_pattern, intent_text_w_markup)

                    curr_char_ix = 0
                    intent_text_without_markup = ''
                    cleaned_text_slots = []  # intent text can contain slots highlighted
                    for line_slot in line_slots_found:
                        line_slot_l_span, line_slot_r_span = line_slot.span()
                        # intent w.o. markup for "some [entity](entity_example) text" is "some entity text"
                        # so we should remove brackets and the parentheses content
                        intent_text_without_markup += intent_text_w_markup[curr_char_ix:line_slot_l_span]

                        slot_value_text = str(line_slot["slot_value"])
                        slot_name = line_slot["slot_name"]
                        slot_value = slot_value_text
                        if ':' in slot_name:
                            slot_name, slot_value = slot_name.split(':', 1)  # e.g. [moderately](price:moderate)

                        assert slot_name in domain_knowledge.known_slots, f"{slot_name} from {nlu_fpath}" + \
                                                                          " was not listed as slot " + \
                                                                          "in domain knowledge config"

                        slot_value_new_l_span = len(intent_text_without_markup)  # l span in cleaned text
                        slot_value_new_r_span = slot_value_new_l_span + len(slot_value_text)  # r span in cleaned text
                        # intent w.o. markup for "some [entity](entity_example) text" is "some entity text"
                        # so we should remove brackets and the parentheses content
                        intent_text_without_markup += slot_value_text

                        cleaned_text_slots.append({"slot_value": slot_value,
                                                   "slot_text": slot_value_text,
                                                   "slot_name": slot_name,
                                                   "span": (slot_value_new_l_span, slot_value_new_r_span)})

                        slot_name2text2value[slot_name][slot_value_text] = slot_value

                        curr_char_ix = line_slot_r_span
                    intent_text_without_markup += intent_text_w_markup[curr_char_ix: len(intent_text_w_markup)]

                    slots_key = tuple(sorted((slot["slot_name"], slot["slot_value"]) for slot in cleaned_text_slots))
                    intent2slots2text[curr_intent_name][slots_key].append({"text": intent_text_without_markup,
                                                                           "slots": cleaned_text_slots})

        # defaultdict behavior is no more needed
        intent2slots2text = {k: dict(v) for k, v in intent2slots2text.items()}
        slot_name2text2value = dict(slot_name2text2value)

        return intent2slots2text, slot_name2text2value

    @classmethod
    def _read_story(cls,
                    story_fpath: Path,
                    dialogs: bool,
                    domain_knowledge: DomainKnowledge,
                    intent2slots2text: Dict[str, Dict[SLOT2VALUE_PAIRS_TUPLE, List]],
                    slot_name2text2value: Dict[str, Dict[str, str]]) \
            -> Union[List[List[Tuple[Dict[str, bool], Dict[str, Any]]]], List[Tuple[Dict[str, bool], Dict[str, Any]]]]:
        """
        Reads stories from the specified path converting them to go-bot format on the fly.

        Args:
            story_fpath: path to the file containing the stories dataset
            dialogs: flag which indicates whether to output list of turns or
                list of dialogs
            domain_knowledge: the domain knowledge, usually inferred from domain.yml
            intent2slots2text: the mapping allowing given the intent class and
                slotfilling values of utterance, restore utterance text.
            slot_name2text2value: the mapping of possible slot values spellings to the values themselves.
        Returns:
            stories read as if it was done with DSTC2DatasetReader._read_from_file()
        """
        log.debug(f"BEFORE MLU_MD_DialogsDatasetReader._read_story(): "
                  f"story_fpath={story_fpath}, "
                  f"dialogs={dialogs}, "
                  f"domain_knowledge={domain_knowledge}, "
                  f"intent2slots2text={intent2slots2text}, "
                  f"slot_name2text2value={slot_name2text2value}")

        default_system_start = {
            "speaker": cls._SYSTEM_SPEAKER_ID,
            "text": "start",
            "dialog_acts": [{"act": "start", "slots": []}]}
        default_system_goodbye = {
            "text": "goodbye :(",
            "dialog_acts": [{"act": "utter_goodbye", "slots": []}],
            "speaker": cls._SYSTEM_SPEAKER_ID}  # TODO infer from dataset

        stories_parsed = {}

        curr_story_title = None
        curr_story_utters = None
        curr_story_bad = False
        for line in open(story_fpath):
            line = line.strip()
            if line.startswith('#'):
                # #... marks the beginning of new story
                if curr_story_utters and curr_story_utters[-1]["speaker"] == cls._USER_SPEAKER_ID:
                    curr_story_utters.append(default_system_goodbye)  # dialogs MUST end with system replics

                if not curr_story_bad:
                    stories_parsed[curr_story_title] = curr_story_utters

                curr_story_title = line.strip('#')
                curr_story_utters = []
                curr_story_bad = False
            elif line.startswith('*'):
                # user actions are started in dataset with *
                user_action, slots_dstc2formatted = cls._parse_user_intent(line)
                slots_actual_values = cls._clarify_slots_values(slot_name2text2value, slots_dstc2formatted)
                try:
                    slots_to_exclude, slots_used_values, action_for_text = cls._choose_slots_for_whom_exists_text(
                        intent2slots2text, slots_actual_values,
                        user_action)
                except KeyError as e:
                    log.debug(f"INSIDE MLU_MD_DialogsDatasetReader._read_story(): "
                              f"Skipping story w. line {line} because of no NLU candidates found")
                    curr_story_bad = True
                    continue
                user_response_info = cls._user_action2text(intent2slots2text, action_for_text, slots_used_values)
                user_utter = {"speaker": cls._USER_SPEAKER_ID,
                              "text": user_response_info["text"],
                              "dialog_acts": [{"act": user_action, "slots": user_response_info["slots"]}],
                              "slots to exclude": slots_to_exclude}

                if not curr_story_utters:
                    curr_story_utters.append(default_system_start)  # dialogs MUST start with system replics
                curr_story_utters.append(user_utter)
            elif line.startswith('-'):
                # system actions are started in dataset with -

                system_action_name = line.strip('-').strip()
                curr_action_text = cls._system_action2text(domain_knowledge, system_action_name)
                system_action = {"speaker": cls._SYSTEM_SPEAKER_ID,
                                 "text": curr_action_text,
                                 "dialog_acts": [{"act": system_action_name, "slots": []}]}
                if system_action_name.startswith("action"):
                    system_action["db_result"] = {}

                if curr_story_utters and curr_story_utters[-1]["speaker"] == cls._SYSTEM_SPEAKER_ID:
                    # deal with consecutive system actions by inserting the last user replics in between
                    last_user_utter = [u for u in reversed(curr_story_utters)
                                       if u["speaker"] == cls._USER_SPEAKER_ID][0]
                    curr_story_utters.append(last_user_utter)

                curr_story_utters.append(system_action)

        if not curr_story_bad:
            stories_parsed[curr_story_title] = curr_story_utters
        stories_parsed.pop(None)

        tmp_f = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding="utf-8")
        for story_id, story in stories_parsed.items():
            for replics in story:
                print(json.dumps(replics), file=tmp_f)
            print(file=tmp_f)
        tmp_f.close()
        # noinspection PyProtectedMember
        gobot_formatted_stories = DSTC2DatasetReader._read_from_file(tmp_f.name, dialogs=dialogs)
        os.remove(tmp_f.name)

        log.debug(f"AFTER MLU_MD_DialogsDatasetReader._read_story(): "
                  f"story_fpath={story_fpath}, "
                  f"dialogs={dialogs}, "
                  f"domain_knowledge={domain_knowledge}, "
                  f"intent2slots2text={intent2slots2text}, "
                  f"slot_name2text2value={slot_name2text2value}")

        return gobot_formatted_stories

    @staticmethod
    def _choose_slots_for_whom_exists_text(intent2slots2text: Dict[str, Dict[SLOT2VALUE_PAIRS_TUPLE, List]],
                                           slots_actual_values: SLOT2VALUE_PAIRS_TUPLE,
                                           user_action: str) -> Tuple[List, SLOT2VALUE_PAIRS_TUPLE, str]:
        possible_keys = [k for k in intent2slots2text.keys() if user_action in k]
        possible_keys = possible_keys + [user_action]
        possible_keys = sorted(possible_keys, key=lambda action_s: action_s.count('+'))
        for possible_action_key in possible_keys:
            if intent2slots2text[possible_action_key].get(slots_actual_values):
                slots_used_values = slots_actual_values
                slots_to_exclude = []
                return slots_to_exclude, slots_used_values, possible_action_key
            else:
                slots_lazy_key = set(e[0] for e in slots_actual_values)
                slots_lazy_key -= {"intent"}
                fake_keys = []
                for known_key in intent2slots2text[possible_action_key].keys():
                    if slots_lazy_key.issubset(set(e[0] for e in known_key)):
                        fake_keys.append(known_key)
                        break

                if fake_keys:
                    slots_used_values = sorted(fake_keys, key=lambda elem: (len(set(slots_actual_values) ^ set(elem)),
                                                                            len([e for e in elem
                                                                                 if e[0] not in slots_lazy_key]))
                                               )[0]

                    slots_to_exclude = [e[0] for e in slots_used_values if e[0] not in slots_lazy_key]
                    return slots_to_exclude, slots_used_values, possible_action_key

        raise KeyError("no possible NLU candidates found")

    @staticmethod
    def _clarify_slots_values(slot_name2text2value: Dict[str, Dict[str, Any]],
                              slots_dstc2formatted: List[List]) -> SLOT2VALUE_PAIRS_TUPLE:
        slots_key = []
        for slot_name, slot_value in slots_dstc2formatted:
            slot_actual_value = slot_name2text2value.get(slot_name, {}).get(slot_value, slot_value)
            slots_key.append((slot_name, slot_actual_value))
        slots_key = tuple(sorted(slots_key))
        return slots_key

    @staticmethod
    def _parse_user_intent(line: str) -> Tuple[str, List[List]]:
        intent = line.strip('*').strip()
        if '{' not in intent:
            intent = intent + "{}"  # the prototypical intent is "intent_name{slot1: value1, slotN: valueN}"
        user_action, slots_info = intent.split('{', 1)
        slots_info = json.loads('{' + slots_info)
        slots_dstc2formatted = [[slot_name, slot_value] for slot_name, slot_value in slots_info.items()]
        return user_action, slots_dstc2formatted

    @staticmethod
    def _user_action2text(intent2slots2text: Dict[str, Dict[SLOT2VALUE_PAIRS_TUPLE, List]],
                          user_action: str,
                          slots_li: Optional[SLOT2VALUE_PAIRS_TUPLE] = None) -> str:
        if slots_li is None:
            slots_li = tuple()
        return intent2slots2text[user_action][slots_li][0]

    @staticmethod
    def _system_action2text(domain_knowledge: DomainKnowledge, system_action: str) -> str:
        possible_system_responses = domain_knowledge.response_templates.get(system_action,
                                                                            [{"text": system_action}])

        response_text = possible_system_responses[0]["text"]
        response_text = re.sub(r"(\w+)\=\{(.*?)\}", r"#\2", response_text)  # TODO: straightforward regex string

        return response_text
