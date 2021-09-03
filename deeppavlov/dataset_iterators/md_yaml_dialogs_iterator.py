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
import itertools
import json
import os
import re
import tempfile
from logging import getLogger
from typing import Dict, List, Tuple, Any, Iterator

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.dataset_readers.dto.rasa.domain_knowledge import DomainKnowledge
from deeppavlov.dataset_readers.dto.rasa.stories import Story, Turn, Stories
from deeppavlov.dataset_readers.dto.rasa.nlu import Intents
import random

log = getLogger(__name__)


class RASADict(dict):
    def __add__(self, oth):
        return RASADict()


@register('md_yaml_dialogs_iterator')
class MD_YAML_DialogsDatasetIterator(DataLearningIterator):
    """this class handles the dialogues provided in form of RASA-like configs"""

    def __init__(self,
                 data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None,
                 shuffle: bool = True,
                 limit: int = 10) -> None:
        self.limit = limit
        super().__init__(data, seed, shuffle)

    def gen_batches(self,
                    batch_size: int,
                    data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple]:
        if shuffle is None:
            shuffle = self.shuffle

        data = self.data[data_type]
        domain_knowledge = self.data[data_type]["domain"]
        intents = self.data[data_type]["nlu_lines"]
        stories = self.data[data_type]["story_lines"]

        dialogs = False
        ignore_slots = False
        story_iterator = StoriesGenerator(stories,
                                          intents,
                                          domain_knowledge,
                                          ignore_slots,
                                          batch_size)

        for batch in story_iterator.generate():
            stories_parsed = batch
            gobot_formatted_stories = DSTC2DatasetReader._read_from_batch(
                list(itertools.chain(*[v + [{}] for v in batch.values()])),
                dialogs=dialogs)

            ds = []
            prev_resp_act = None
            for x, y in gobot_formatted_stories:
                if x.get('episode_done'):
                    del x['episode_done']
                    prev_resp_act = None
                    ds.append(([], []))
                x['prev_resp_act'] = prev_resp_act
                prev_resp_act = y['act']
                ds[-1][0].append(x)
                ds[-1][1].append(y)
            yield zip(*ds)

    def get_instances(self, data_type: str = 'train') -> Tuple[
        tuple, tuple]:
        concat = lambda it: tuple(itertools.chain(*it))
        tmp = self.gen_batches(batch_size=-1,
                               data_type=data_type,
                               shuffle=False)
        res = tuple(e for el in tmp
                    for e in el)
        return res


class TurnIterator:
    """this class handles the stories turns in RASA"""
    _USER_SPEAKER_ID = 1
    _SYSTEM_SPEAKER_ID = 2

    def __init__(self, turn: Turn, nlu: Intents,
                 domain_knowledge: DomainKnowledge, ignore_slots: bool = False):
        self.turn = turn
        self.intents: Intents = nlu
        self.domain_knowledge = domain_knowledge
        self.ignore_slots = ignore_slots

    def _clarify_slots_values(self, slots_dstc2formatted):
        slots_key = []
        for slot_name, slot_value in slots_dstc2formatted:
            slot_actual_value = self.intents.slot_name2text2value.get(slot_name,
                                                                      {}).get(
                slot_value, slot_value)
            slots_key.append((slot_name, slot_actual_value))
        slots_key = tuple(sorted(slots_key))
        return slots_key

    def parse_user_intent(self):
        """
                Given the intent line in RASA stories.md format, return the name of the intent and slots described with this line
                Args:
                    line: the line to parse
                Returns:
                    the pair of the intent name and slots ([[slot name, slot value],.. ]) info
                """
        intent = self.turn.turn_description.strip('*').strip()
        if '{' not in intent:
            intent = intent + "{}"  # the prototypical intent is "intent_name{slot1: value1, slotN: valueN}"
        user_action, slots_info = intent.split('{', 1)
        slots_info = json.loads('{' + slots_info)
        slots_dstc2formatted = [[slot_name, slot_value] for
                                slot_name, slot_value in slots_info.items()]
        if self.ignore_slots:
            slots_dstc2formatted = dict()
        return user_action, slots_dstc2formatted

    def choose_slots_for_whom_exists_text(self, slots_actual_values,
                                          user_action):
        """
                Args:
                    slots_actual_values: the slot values information to look utterance for
                    user_action: the intent to look utterance for
                Returns:
                    the slots ommitted to find an NLU candidate, the slots represented in the candidate, the intent name used
                """
        possible_keys = [k for k in self.intents.intent2slots2text.keys() if
                         user_action in k]
        possible_keys = possible_keys + [user_action]
        possible_keys = sorted(possible_keys,
                               key=lambda action_s: action_s.count('+'))
        for possible_action_key in possible_keys:
            if self.intents.intent2slots2text[possible_action_key].get(
                    slots_actual_values):
                slots_used_values = slots_actual_values
                slots_to_exclude = []
                return slots_to_exclude, slots_used_values, possible_action_key
            else:
                slots_lazy_key = set(e[0] for e in slots_actual_values)
                slots_lazy_key -= {"intent"}
                fake_keys = []
                for known_key in self.intents.intent2slots2text[
                    possible_action_key].keys():
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

    def user_action2text(self, user_action: str, slots_li=None):
        """
        given the user intent, return the text representing this intent with passed slots
        Args:
            user_action: the name of intent to generate text for
            slots_li: the slot values to provide
        Returns:
            the text of utterance relevant to the passed intent and slots
        """
        if slots_li is None:
            slots_li = tuple()
        res = self.intents.intent2slots2text[user_action][slots_li]
        return res

    def process_user_turn(self):
        user_action, slots_dstc2formatted = self.parse_user_intent()
        slots_actual_values = self._clarify_slots_values(slots_dstc2formatted)
        slots_to_exclude, slots_used_values, action_for_text = self.choose_slots_for_whom_exists_text(
            slots_actual_values, user_action)
        possible_user_response_infos = self.user_action2text(action_for_text,
                                                             slots_used_values)
        random.shuffle(possible_user_response_infos)

        for user_response_info in possible_user_response_infos[:2]:
            user_utter = {"speaker": self._USER_SPEAKER_ID,
                          "text": user_response_info["text"],
                          "dialog_acts": [{"act": user_action,
                                           "slots": user_response_info[
                                               "slots"]}],
                          "slots to exclude": slots_to_exclude}
            yield user_utter

    def system_action2text(self, system_action):
        """
                given the system action name return the relevant template text
                Args:
                    domain_knowledge: the domain knowledge relevant to the currently processed config
                    system_action: the name of the action to get intent for
                Returns:
                    template relevant to the passed action
                """
        possible_system_responses = self.domain_knowledge.response_templates.get(
            system_action,
            [{"text": system_action}])

        response_text = possible_system_responses[0]["text"]
        response_text = re.sub(r"(\w+)\=\{(.*?)\}", r"#\2",
                               response_text)  # TODO: straightforward regex string

        return response_text

    def parse_system_turn(self):
        """
                Given the RASA stories.md line, returns the dstc2-formatted json (dict) for this line
                Args:
                    domain_knowledge: the domain knowledge relevant to the processed stories config (from which line is taken)
                    line: the story system step representing line from stories.md
                Returns:
                    the dstc2-formatted passed turn
                """

        system_action_name = self.turn.turn_description.strip('-').strip()
        curr_action_text = self.system_action2text(system_action_name)
        system_action = {"speaker": self._SYSTEM_SPEAKER_ID,
                         "text": curr_action_text,
                         "dialog_acts": [
                             {"act": system_action_name, "slots": []}]}
        if system_action_name.startswith("action"):
            system_action["db_result"] = {}
        return system_action

    def process_system_utter(self):
        """
        Yields: all the possible dstc2 versions of the passed story line
        TODO: SUPPORT FORMS
        """
        system_action = self.parse_system_turn()
        yield system_action

    def __call__(self):
        if self.turn.is_user_turn():
            for possible_turn in self.process_user_turn():
                yield possible_turn
        elif self.turn.is_system_turn():
            for possible_turn in self.process_system_utter():
                yield possible_turn


def iterProduct(ic):
    # https://stackoverflow.com/a/12094245
    if not ic:
        yield []
        return

    for i in ic[0]():
        for js in iterProduct(ic[1:]):
            yield [i] + js


class StoryGenerator:
    def __init__(self, story: Story, nlu: Intents,
                 domain_knowledge: DomainKnowledge, ignore_slots=False):
        self.story: Story = story
        self.turn_iterators = []
        for turn in story.turns:
            turn_iterator = TurnIterator(turn, nlu, domain_knowledge,
                                         ignore_slots)
            self.turn_iterators.append(turn_iterator)
        self.turn_ix = -1
        self.version_ix = -1

    def gen_story_sample(self):
        for i in iterProduct(self.turn_iterators):
            yield i


class StoriesGenerator:
    """this class handles stories defined in RASA configs"""
    def __init__(self, stories: Stories, intents: Intents,
                 domain_knowledge: DomainKnowledge, ignore_slots: False,
                 batch_size=1):
        self.stories = stories
        self.intents = intents
        self.domain_knowledge = domain_knowledge
        self.ignore_slots = ignore_slots
        self.batch_size = batch_size

    def generate(self):
        batch = dict()
        for story in self.stories.stories:
            story_generator = StoryGenerator(story, self.intents,
                                             self.domain_knowledge,
                                             self.ignore_slots)
            for story_data in story_generator.gen_story_sample():
                batch[story.title] = story_data
                if len(batch) == self.batch_size:
                    yield batch
                    batch = dict()
        yield batch
