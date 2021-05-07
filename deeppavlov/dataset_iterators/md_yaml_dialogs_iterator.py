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
import json
import os
import re
import tempfile
from collections import defaultdict
from logging import getLogger
from typing import Tuple, List, Dict, Any, Iterator, Union, Optional

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.dataset_readers.dto.rasa.domain_knowledge import DomainKnowledge

log = getLogger(__name__)

SLOT2VALUE_PAIRS_TUPLE = Tuple[Tuple[str, Any], ...]



"""

"""




@register('md_yaml_dialogs_iterator')
class MD_YAML_DialogsDatasetIterator(DataLearningIterator):
    def __init__(self,
                 data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None,
                 shuffle: bool = True) -> None:
        super().__init__(data, seed, shuffle)

    def gen_batches(self,
                    batch_size: int,
                    data_type: str = 'train',
                    shuffle: bool = None,
                    ignore_slots: bool = False,
                    dialogs: bool = False) -> Iterator[Tuple]:

        data = self.data[data_type]
        story_lines = data["story_lines"]
        domain = data["domain"]
        nlu_lines = data["nlu_lines"]

        intent2slots2text, slot_name2text2value = self._read_nlu(
            domain,
            ignore_slots,
            nlu_lines)
        s = self._read_story(story_lines,
                            dialogs, domain,
                            intent2slots2text, slot_name2text2value,
                            ignore_slots=ignore_slots)
        pass

    @classmethod
    def _read_nlu(cls, domain_knowledge, ignore_slots, nlu_lines):
        slots_markup_pattern = r"\[" + \
                               r"(?P<slot_value>.*?)" + \
                               r"\]" + \
                               r"\(" + \
                               r"(?P<slot_name>.*?)" + \
                               r"\)"

        intent2slots2text = defaultdict(lambda: defaultdict(list))
        slot_name2text2value = defaultdict(lambda: defaultdict(list))
        curr_intent_name = None
        for line in nlu_lines:
            if line.startswith("##"):
                # lines starting with ## are starting section describing new intent type
                curr_intent_name = line.strip("##").strip().split("intent:", 1)[-1]
            if line.strip().startswith('-'):
                # lines starting with - are listing the examples of intent texts of the current intent type
                intent_text_w_markup = line.strip().strip('-').strip()
                line_slots_found = re.finditer(slots_markup_pattern, intent_text_w_markup)
                if ignore_slots:
                    line_slots_found = []

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
                        # e.g. [moderately](price:moderate)
                        slot_name, slot_value = slot_name.split(':', 1)

                    assert slot_name in domain_knowledge.known_slots, f"{slot_name}" + \
                                                                      " was not listed as slot " + \
                                                                      "in domain knowledge config"

                    slot_value_new_l_span = len(intent_text_without_markup)  # l span in cleaned text
                    slot_value_new_r_span = slot_value_new_l_span + len(slot_value_text)  # r span in cleaned text
                    # intent w.o. markup for "some [entity](entity_example) text" is "some entity text"
                    # so we should remove brackets and the parentheses content
                    intent_text_without_markup += slot_value_text

                    cleaned_text_slots.append((slot_name, slot_value))

                    slot_name2text2value[slot_name][slot_value_text].append(slot_value)

                    curr_char_ix = line_slot_r_span
                intent_text_without_markup += intent_text_w_markup[curr_char_ix: len(intent_text_w_markup)]

                slots_key = tuple(sorted((slot[0], slot[1]) for slot in cleaned_text_slots))
                intent2slots2text[curr_intent_name][slots_key].append(
                    {"text": intent_text_without_markup,
                     "slots_di": cleaned_text_slots,
                     "slots": slots_key})
        # defaultdict behavior is no more needed
        intent2slots2text = {k: dict(v) for k, v in intent2slots2text.items()}
        slot_name2text2value = dict(slot_name2text2value)
        return intent2slots2text, slot_name2text2value

    @classmethod
    def _read_story(cls,
                    story_lines: List,
                    dialogs: bool,
                    domain_knowledge: DomainKnowledge,
                    intent2slots2text: Dict[str, Dict[SLOT2VALUE_PAIRS_TUPLE, List]],
                    slot_name2text2value: Dict[str, Dict[str, str]],
                    ignore_slots: bool = False) \
            -> Union[List[List[Tuple[Dict[str, bool], Dict[str, Any]]]], List[
                Tuple[Dict[str, bool], Dict[str, Any]]]]:
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
        curr_story_utters_batch = []
        nonlocal_curr_story_bad = False  # can be modified as a nonlocal variable

        def process_user_utter(line: str) -> List[List[Dict[str, Any]]]:
            """
            given the stories.md user line, returns the batch of all the dstc2 ways to represent it
            Args:
                line: the system line to generate dstc2 versions for

            Returns:
                all the possible dstc2 versions of the passed story line
            """
            nonlocal intent2slots2text, slot_name2text2value, curr_story_utters_batch, nonlocal_curr_story_bad
            try:
                possible_user_utters = cls._augment_user_turn(intent2slots2text,
                                                              line,
                                                              slot_name2text2value)
                # dialogs MUST start with system replics
                for curr_story_utters in curr_story_utters_batch:
                    if not curr_story_utters:
                        curr_story_utters.append(default_system_start)

                utters_to_append_batch = []
                for user_utter in possible_user_utters:
                    utters_to_append_batch.append([user_utter])

            except KeyError:
                log.debug(f"INSIDE MLU_MD_DialogsDatasetReader._read_story(): "
                          f"Skipping story w. line {line} because of no NLU candidates found")
                nonlocal_curr_story_bad = True
                utters_to_append_batch = []
            return utters_to_append_batch

        def process_system_utter(line: str) -> List[List[Dict[str, Any]]]:
            """
            given the stories.md system line, returns the batch of all the dstc2 ways to represent it
            Args:
                line: the system line to generate dstc2 versions for

            Returns:
                all the possible dstc2 versions of the passed story line
            """
            nonlocal intent2slots2text, domain_knowledge, curr_story_utters_batch, nonlocal_curr_story_bad
            system_action = cls._parse_system_turn(domain_knowledge, line)
            system_action_name = system_action.get("dialog_acts")[0].get("act")

            for curr_story_utters in curr_story_utters_batch:
                if cls._last_turn_is_systems_turn(curr_story_utters):
                    # deal with consecutive system actions by inserting the last user replics in between
                    curr_story_utters.append(
                        cls._get_last_users_turn(curr_story_utters))

            def parse_form_name(story_line: str) -> str:
                """
                if the line (in stories.md utterance format) contains a form name, return it
                Args:
                    story_line: line to extract form name from

                Returns:
                    the extracted form name or None if no form name found
                """
                form_name = None
                if story_line.startswith("form"):
                    form_di = json.loads(story_line[len("form"):])
                    form_name = form_di["name"]
                return form_name

            if system_action_name.startswith("form"):
                form_name = parse_form_name(system_action_name)
                augmented_utters = cls._augment_form(form_name, domain_knowledge,
                                                     intent2slots2text)

                utters_to_append_batch = [[]]
                for user_utter in augmented_utters:
                    new_curr_story_utters_batch = []
                    for curr_story_utters in utters_to_append_batch:
                        possible_extensions = process_story_line(user_utter)
                        for possible_extension in possible_extensions:
                            new_curr_story_utters = curr_story_utters.copy()
                            new_curr_story_utters.extend(possible_extension)
                            new_curr_story_utters_batch.append(
                                new_curr_story_utters)
                    utters_to_append_batch = new_curr_story_utters_batch
            else:
                utters_to_append_batch = [[system_action]]
            return utters_to_append_batch

        def process_story_line(line: str) -> List[List[Dict[str, Any]]]:
            """
            given the stories.md line, returns the batch of all the dstc2 ways to represent it
            Args:
                line: the line to generate dstc2 versions

            Returns:
                all the possible dstc2 versions of the passed story line
            """
            if line.startswith('*'):
                utters_to_extend_with_batch = process_user_utter(line)
            elif line.startswith('-'):
                utters_to_extend_with_batch = process_system_utter(line)
            else:
                # todo raise an exception
                utters_to_extend_with_batch = []
            return utters_to_extend_with_batch

        for line in story_lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                # #... marks the beginning of new story
                if curr_story_utters_batch and curr_story_utters_batch[0] and \
                        curr_story_utters_batch[0][-1][
                            "speaker"] == cls._USER_SPEAKER_ID:
                    for curr_story_utters in curr_story_utters_batch:
                        curr_story_utters.append(
                            default_system_goodbye)  # dialogs MUST end with system replics

                if not nonlocal_curr_story_bad:
                    for curr_story_utters_ix, curr_story_utters in enumerate(
                            curr_story_utters_batch):
                        stories_parsed[
                            curr_story_title + f"_{curr_story_utters_ix}"] = curr_story_utters

                curr_story_title = line.strip('#')
                curr_story_utters_batch = [[]]
                nonlocal_curr_story_bad = False
            else:
                new_curr_story_utters_batch = []
                possible_extensions = process_story_line(line)
                for curr_story_utters in curr_story_utters_batch:
                    for user_utter in possible_extensions:
                        new_curr_story_utters = curr_story_utters.copy()
                        new_curr_story_utters.extend(user_utter)
                        new_curr_story_utters_batch.append(
                            new_curr_story_utters)
                curr_story_utters_batch = new_curr_story_utters_batch
                # curr_story_utters.extend(process_story_line(line))

        if not nonlocal_curr_story_bad:
            for curr_story_utters_ix, curr_story_utters in enumerate(
                    curr_story_utters_batch):
                stories_parsed[
                    curr_story_title + f"_{curr_story_utters_ix}"] = curr_story_utters

        tmp_f = tempfile.NamedTemporaryFile(delete=False, mode='w',
                                            encoding="utf-8")
        for story_id, story in stories_parsed.items():
            for replics in story:
                print(json.dumps(replics), file=tmp_f)
            print(file=tmp_f)
        tmp_f.close()
        # noinspection PyProtectedMember
        gobot_formatted_stories = DSTC2DatasetReader._read_from_file(tmp_f.name,
                                                                     dialogs=dialogs)
        os.remove(tmp_f.name)

        log.debug(f"AFTER MLU_MD_DialogsDatasetReader._read_story(): "
                  f"dialogs={dialogs}, "
                  f"domain_knowledge={domain_knowledge}, "
                  f"intent2slots2text={intent2slots2text}, "
                  f"slot_name2text2value={slot_name2text2value}")

        return gobot_formatted_stories

    @classmethod
    def _augment_form(cls, form_name: str, domain_knowledge: DomainKnowledge,
                      intent2slots2text: Dict) -> List[str]:
        """
        Replaced the form mention in stories.md with the actual turns relevant to the form
        Args:
            form_name: the name of form to generate turns for
            domain_knowledge: the domain knowledge (see domain.yml in RASA) relevant to the processed config
            intent2slots2text: the mapping of intents and particular slots onto text

        Returns:
            the story turns relevant to the passed form
        """
        form = domain_knowledge.forms[form_name]  # todo handle keyerr
        augmended_story = []
        for slot_name, slot_info_li in form.items():
            if slot_info_li and slot_info_li[0].get("type",
                                                    '') == "from_entity":
                # we only handle from_entity slots
                known_responses = list(domain_knowledge.response_templates)
                known_intents = list(intent2slots2text.keys())
                augmended_story.extend(
                    cls._augment_slot(known_responses, known_intents, slot_name,
                                      form_name))
        return augmended_story

    @classmethod
    def _augment_slot(cls, known_responses: List[str], known_intents: List[str],
                      slot_name: str, form_name: str) \
            -> List[str]:
        """
        Given the slot name, generates a sequence of system turn asking for a slot and user' turn providing this slot

        Args:
            known_responses: responses known to the system from domain.yml
            known_intents: intents known to the system from domain.yml
            slot_name: the name of the slot to augment for
            form_name: the name of the form for which the turn is augmented

        Returns:
            the list of stories.md alike turns
        """
        ask_slot_act_name = cls._get_augmented_ask_slot_utter(form_name,
                                                              known_responses,
                                                              slot_name)
        inform_slot_user_utter = cls._get_augmented_ask_intent_utter(
            known_intents, slot_name)

        return [f"- {ask_slot_act_name}", f"* {inform_slot_user_utter}"]

    @classmethod
    def _get_augmented_ask_intent_utter(cls, known_intents: List[str],
                                        slot_name: str) -> Optional[str]:
        """
        if the system knows the inform_{slot} intent, return this intent name, otherwise return None
        Args:
            known_intents: intents known to the system
            slot_name: the slot to look inform intent for

        Returns:
            the slot informing intent or None
        """
        inform_slot_user_utter_hypothesis = f"inform_{slot_name}"
        if inform_slot_user_utter_hypothesis in known_intents:
            inform_slot_user_utter = inform_slot_user_utter_hypothesis
        else:
            # todo raise an exception
            inform_slot_user_utter = None
            pass
        return inform_slot_user_utter

    @classmethod
    def _get_augmented_ask_slot_utter(cls, form_name: str,
                                      known_responses: List[str],
                                      slot_name: str):
        """
        if the system knows the ask_{slot} action, return this action name, otherwise return None
        Args:
            form_name: the name of the currently processed form
            known_responses: actions known to the system
            slot_name: the slot to look asking action for

        Returns:
            the slot asking action or None
        """
        ask_slot_act_name_hypothesis1 = f"utter_ask_{form_name}_{slot_name}"
        ask_slot_act_name_hypothesis2 = f"utter_ask_{slot_name}"
        if ask_slot_act_name_hypothesis1 in known_responses:
            ask_slot_act_name = ask_slot_act_name_hypothesis1
        elif ask_slot_act_name_hypothesis2 in known_responses:
            ask_slot_act_name = ask_slot_act_name_hypothesis2
        else:
            # todo raise an exception
            ask_slot_act_name = None
            pass
        return ask_slot_act_name

    @classmethod
    def _get_last_users_turn(cls, curr_story_utters: List[Dict]) -> Dict:
        """
        Given the dstc2 story, return the last user utterance from it
        Args:
            curr_story_utters: the dstc2-formatted stoyr

        Returns:
            the last user utterance from the passed story
        """
        *_, last_user_utter = filter(
            lambda x: x["speaker"] == cls._USER_SPEAKER_ID, curr_story_utters)
        return last_user_utter

    @classmethod
    def _last_turn_is_systems_turn(cls, curr_story_utters):
        return curr_story_utters and curr_story_utters[-1][
            "speaker"] == cls._SYSTEM_SPEAKER_ID

    @classmethod
    def _parse_system_turn(cls, domain_knowledge: DomainKnowledge,
                           line: str) -> Dict:
        """
        Given the RASA stories.md line, returns the dstc2-formatted json (dict) for this line
        Args:
            domain_knowledge: the domain knowledge relevant to the processed stories config (from which line is taken)
            line: the story system step representing line from stories.md

        Returns:
            the dstc2-formatted passed turn
        """
        # system actions are started in dataset with -
        system_action_name = line.strip('-').strip()
        curr_action_text = cls._system_action2text(domain_knowledge,
                                                   system_action_name)
        system_action = {"speaker": cls._SYSTEM_SPEAKER_ID,
                         "text": curr_action_text,
                         "dialog_acts": [
                             {"act": system_action_name, "slots": []}]}
        if system_action_name.startswith("action"):
            system_action["db_result"] = {}
        return system_action

    @classmethod
    def _augment_user_turn(cls, intent2slots2text, line: str,
                           slot_name2text2value) -> List[Dict[str, Any]]:
        """
        given the turn information generate all the possible stories representing it
        Args:
            intent2slots2text: the intents and slots to natural language utterances mapping known to the system
            line: the line representing used utterance in stories.md format
            slot_name2text2value: the slot names to values mapping known o the system

        Returns:
            the batch of all the possible dstc2 representations of the passed intent
        """
        # user actions are started in dataset with *
        user_action, slots_dstc2formatted = cls._parse_user_intent(line)
        slots_actual_values = cls._clarify_slots_values(slot_name2text2value,
                                                        slots_dstc2formatted)
        slots_to_exclude, slots_used_values, action_for_text = cls._choose_slots_for_whom_exists_text(
            intent2slots2text, slots_actual_values,
            user_action)
        possible_user_response_infos = cls._user_action2text(intent2slots2text,
                                                             action_for_text,
                                                             slots_used_values)
        possible_user_utters = []
        for user_response_info in possible_user_response_infos:
            user_utter = {"speaker": cls._USER_SPEAKER_ID,
                          "text": user_response_info["text"],
                          "dialog_acts": [{"act": user_action,
                                           "slots": user_response_info[
                                               "slots"]}],
                          "slots to exclude": slots_to_exclude}
            possible_user_utters.append(user_utter)
        return possible_user_utters

    @staticmethod
    def _choose_slots_for_whom_exists_text(
            intent2slots2text: Dict[str, Dict[SLOT2VALUE_PAIRS_TUPLE, List]],
            slots_actual_values: SLOT2VALUE_PAIRS_TUPLE,
            user_action: str) -> Tuple[List, SLOT2VALUE_PAIRS_TUPLE, str]:
        """

        Args:
            intent2slots2text: the mapping of intents and slots to natural language utterances representing them
            slots_actual_values: the slot values information to look utterance for
            user_action: the intent to look utterance for

        Returns:
            the slots ommitted to find an NLU candidate, the slots represented in the candidate, the intent name used
        """
        possible_keys = [k for k in intent2slots2text.keys() if
                         user_action in k]
        possible_keys = possible_keys + [user_action]
        possible_keys = sorted(possible_keys,
                               key=lambda action_s: action_s.count('+'))
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
                    slots_used_values = sorted(fake_keys, key=lambda elem: (
                    len(set(slots_actual_values) ^ set(elem)),
                    len([e for e in elem
                         if e[0] not in slots_lazy_key]))
                                               )[0]

                    slots_to_exclude = [e[0] for e in slots_used_values if
                                        e[0] not in slots_lazy_key]
                    return slots_to_exclude, slots_used_values, possible_action_key

        raise KeyError("no possible NLU candidates found")

    @staticmethod
    def _clarify_slots_values(slot_name2text2value: Dict[str, Dict[str, Any]],
                              slots_dstc2formatted: List[
                                  List]) -> SLOT2VALUE_PAIRS_TUPLE:
        slots_key = []
        for slot_name, slot_value in slots_dstc2formatted:
            slot_actual_value = slot_name2text2value.get(slot_name, {}).get(
                slot_value, slot_value)
            slots_key.append((slot_name, slot_actual_value))
        slots_key = tuple(sorted(slots_key))
        return slots_key

    @staticmethod
    def _parse_user_intent(line: str, ignore_slots=False) -> Tuple[
        str, List[List]]:
        """
        Given the intent line in RASA stories.md format, return the name of the intent and slots described with this line
        Args:
            line: the line to parse
            ignore_slots: whether to ignore slots information

        Returns:
            the pair of the intent name and slots ([[slot name, slot value],.. ]) info
        """
        intent = line.strip('*').strip()
        if '{' not in intent:
            intent = intent + "{}"  # the prototypical intent is "intent_name{slot1: value1, slotN: valueN}"
        user_action, slots_info = intent.split('{', 1)
        slots_info = json.loads('{' + slots_info)
        slots_dstc2formatted = [[slot_name, slot_value] for
                                slot_name, slot_value in slots_info.items()]
        if ignore_slots:
            slots_dstc2formatted = dict()
        return user_action, slots_dstc2formatted

    @staticmethod
    def _user_action2text(
            intent2slots2text: Dict[str, Dict[SLOT2VALUE_PAIRS_TUPLE, List]],
            user_action: str,
            slots_li: Optional[SLOT2VALUE_PAIRS_TUPLE] = None) -> List[str]:
        """
        given the user intent, return the text representing this intent with passed slots
        Args:
            intent2slots2text: the mapping of intents and slots to natural language utterances
            user_action: the name of intent to generate text for
            slots_li: the slot values to provide

        Returns:
            the text of utterance relevant to the passed intent and slots
        """
        if slots_li is None:
            slots_li = tuple()
        return intent2slots2text[user_action][slots_li]

    @staticmethod
    def _system_action2text(domain_knowledge: DomainKnowledge,
                            system_action: str) -> str:
        """
        given the system action name return the relevant template text
        Args:
            domain_knowledge: the domain knowledge relevant to the currently processed config
            system_action: the name of the action to get intent for

        Returns:
            template relevant to the passed action
        """
        possible_system_responses = domain_knowledge.response_templates.get(
            system_action,
            [{"text": system_action}])

        response_text = possible_system_responses[0]["text"]
        response_text = re.sub(r"(\w+)\=\{(.*?)\}", r"#\2",
                               response_text)  # TODO: straightforward regex string

        return response_text
