import json
from pathlib import Path
from typing import List, Iterator, Union, Optional, Dict, Tuple

import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_yaml
from deeppavlov.core.common.registry import register
from deeppavlov.dataset_readers.md_yaml_dialogs_reader import DomainKnowledge, MD_YAML_DialogsDatasetReader
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
        actions_required_acquired_slots_path: (optional) path to json-file with mapping
            of actions to slots that should be filled to allow for action to be executed
    """

    def get_current_knowledge(self) -> TrackerKnowledgeInterface:
        raise NotImplementedError("Featurized tracker lacks get_current_knowledge() method. "
                                  "To be improved in future versions.")

    def __init__(self,
                 slot_names: List[str],
                 # actions_required_acquired_slots_path: Optional[Union[str, Path]]=None,
                 domain_yml_path: Optional[Union[str, Path]]=None,
                 stories_yml_path: Optional[Union[str, Path]]=None,
                 **kwargs) -> None:
        self.slot_names = list(slot_names)
        self.domain_yml_path = domain_yml_path
        self.stories_path = stories_yml_path
        self.action_names2required_slots, self.action_names2acquired_slots =\
            self._load_actions2slots_formfilling_info_from(domain_yml_path, stories_yml_path)
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

    def _load_actions2slots_formfilling_info_from_json(self,
                                                       actions_required_acquired_slots_path: Optional[Union[str, Path]] = None)\
            -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        loads the formfilling mapping of actions onto the required slots from the json of the following structure:
        {action1: {"required": [required_slot_name_1], "acquired": [acquired_slot_name_1, acquired_slot_name_2]},
         action2: {"required": [required_slot_name_21, required_slot_name_22], "acquired": [acquired_slot_name_21]},
        ..}
        Returns:
             the dictionary represented by the passed json
        """
        actions_required_acquired_slots_path = expand_path(actions_required_acquired_slots_path)
        with open(actions_required_acquired_slots_path, encoding="utf-8") as actions2slots_json_f:
            actions2slots = json.load(actions2slots_json_f)
            actions2required_slots = {act: act_slots["required"] for act, act_slots in actions2slots.items()}
            actions2acquired_slots = {act: act_slots["acquired"] for act, act_slots in actions2slots.items()}
        return actions2required_slots, actions2acquired_slots

    def _load_actions2slots_formfilling_info_from(self,
                                                  domain_yml_path: Optional[Union[str, Path]],
                                                  stories_yml_path: Optional[Union[str, Path]])\
            -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        loads the formfilling mapping of actions onto the required slots from the domain.yml form description:

            restaurant_form:
                cuisine:
                  - type: from_entity
                    entity: cuisine
                num_people:
                  - type: from_entity
                    entity: number

        Returns:
             the dictionary represented by the passed json
        """
        if domain_yml_path is None or stories_yml_path is None:
            return {}, {}

        domain_yml_path = expand_path(domain_yml_path)
        domain_knowledge: DomainKnowledge = DomainKnowledge.from_yaml(domain_yml_path)
        potential_api_or_db_actions = domain_knowledge.known_actions
        forms = domain_knowledge.forms
        form_names = list(forms.keys())

        # todo migrate to rasa2.0
        def read_md_story(story_path: Union[Path, str]) -> Dict[str, List[Dict]]:
            """
            given the path to stories.md naively read steps from it. ToDo use MDYAML reader
            Args:
                story_path: the path to stories.md

            Returns:
                the dict containing info on all the stories used
            """
            story_f = open(story_path, 'r')
            stories_li = []
            curr_story = None
            for line in story_f:
                line = line.strip()
                if not line: continue;
                if line.startswith("#"):
                    if curr_story is not None:
                        stories_li.append(curr_story)
                    story_name = line.strip('#').strip()
                    curr_story = {"story": story_name, "steps": []}
                elif line.startswith("*"):
                    # user turn
                    step = {"intent": line.strip('*').strip()}
                    curr_story["steps"].append(step)
                elif line.startswith('-'):
                    # system turn
                    step = {"action": line.strip('-').strip()}
                    curr_story["steps"].append(step)
            if curr_story is not None:
                stories_li.append(curr_story)
            story_f.close()
            stories_di = {"stories": stories_li}
            return stories_di

        stories_md_path = expand_path(stories_yml_path)
        stories_yml_di = read_md_story(stories_md_path)
        prev_forms = []
        action2forms = {}
        for story in stories_yml_di["stories"]:
            story_name = story["story"]
            story_steps = story["steps"]
            for step in story_steps:
                if "action" not in step.keys():
                    continue

                curr_action = step["action"]
                if curr_action.startswith("form"):
                    curr_action = json.loads(curr_action[len("form"):])["name"]
                    print(curr_action)
                if curr_action in form_names:
                    prev_forms.append(curr_action)
                if curr_action in potential_api_or_db_actions:
                    action2forms[curr_action] = prev_forms
                    prev_forms = []

        def get_slots(system_utter: str, form_name: str) -> List[str]:
            """
            Given the utterance story line, extract slots information from it
            Args:
                system_utter: the utterance story line
                form_name: the form we are filling

            Returns:
                the slots extracted from the line
            """
            slots = []
            if system_utter.startswith(f"utter_ask_{form_name}_"):
                slots.append(system_utter[len(f"utter_ask_{form_name}_"):])
            elif system_utter.startswith(f"utter_ask_"):
                slots.append(system_utter[len(f"utter_ask_"):])
            else:
                # todo: raise an exception
                pass
            return slots

        actions2acquired_slots = {utter.strip('-').strip(): get_slots(utter.strip('-').strip(), form_name)
                                  for form_name, form in forms.items()
                                  for utter in
                                  MD_YAML_DialogsDatasetReader.augment_form(form_name, domain_knowledge, {})
                                  if utter.strip().startswith("-")}
        forms2acquired_slots = {form_name: self._get_form_acquired_slots(form) for form_name, form in forms.items()}
        actions2required_slots = {act: {slot
                                        for form in forms
                                        for slot in forms2acquired_slots[form]}
                                  for act, forms in action2forms.items()}
        return actions2required_slots, actions2acquired_slots

    def _get_form_acquired_slots(self, form: Dict) -> List[str]:
        """
        given the form, return the slots that are acquired with this form
        Args:
            form: form to extract acquired slots from

        Returns:
            the slots acquired from the passed form
        """
        acquired_slots = [slot_name
                          for slot_name, slot_info_li in form.items()
                          if slot_info_li and slot_info_li[0].get("type", '') == "from_entity"]
        return acquired_slots
