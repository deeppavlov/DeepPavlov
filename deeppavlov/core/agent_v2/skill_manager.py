import copy
import operator
from itertools import compress
from typing import List, Dict, Optional, Sequence

from deeppavlov.core.agent_v2.config import MAX_WORKERS, SKILLS
from deeppavlov.core.agent_v2.hardcode_utterances import NOANSWER_UTT
from deeppavlov.core.agent_v2.state_schema import Human


class SkillManager:

    def __init__(self, response_selector, skill_caller, skill_selector=None, profile_handlers: Sequence[str] = ()):
        """

        Args:
            response_selector:
            skill_caller:
            skill_selector:
            profile_handlers: list of skill names for getting user profiles info ordered from most to least important
                names not in ``self.skill_names`` are ignored
        """
        self.skill_selector = skill_selector
        self.response_selector = response_selector
        self.max_workers = MAX_WORKERS
        self.skill_caller = skill_caller
        self.skills = SKILLS
        self.skill_names = [s['name'] for s in self.skills]
        self.skill_responses = []

        self.profile_handlers = [name for name in reversed(profile_handlers) if name in self.skill_names]
        self.profile_fields = list(Human.profile.default.keys())

    def __call__(self, state):

        user_profiles = self._get_user_profiles(self.skill_responses, state['dialogs'])
        selected_skill_names, utterances, confidences = self.response_selector(self.skill_responses, state)
        utterances = [utt if utt else NOANSWER_UTT for utt in utterances]
        return selected_skill_names, utterances, confidences, user_profiles

    def _get_user_profiles(self, skill_responses, dialogs) -> Optional[List[Dict]]:
        """
        Get user profile descriptors from compatible skills.
        """
        user_profiles = []
        for sr, dialog in zip(skill_responses, dialogs):
            profile = {}
            for item in self.profile_fields:
                for skill in self.profile_handlers:
                    try:
                        profile[item] = sr[skill][item]
                    except KeyError:
                        pass
            user_profiles.append(profile)
        return user_profiles

    def get_skill_responses(self, state):
        dialogs = state['dialogs']
        n_dialogs = len(dialogs)
        skill_names = [s['name'] for s in self.skills]
        skill_urls = [s['url'] for s in self.skills]
        if self.skill_selector is not None:
            selected_skills = self.skill_selector(state)
        else:
            selected_skills = skill_names * n_dialogs

        excluded_skills = []
        for active_names in selected_skills:
            excluded_skills.append([n not in active_names for n in skill_names])
        excluded_skills = list(map(list, zip(*excluded_skills)))

        payloads = []
        for exclude, skill in zip(excluded_skills, self.skills):
            s = copy.deepcopy(state)
            compressed_dialogs = list(compress(s['dialogs'], map(operator.not_, exclude)))
            if not compressed_dialogs:
                skill_names.remove(skill['name'])
                skill_urls.remove(skill['url'])
                continue
            s['dialogs'] = compressed_dialogs
            payloads.append(s)

        skill_responses = self.skill_caller(payload=payloads, names=skill_names, urls=skill_urls)
        for response, dialog in zip(skill_responses, dialogs):
            # if 'hellobot' in response and len(dialog['utterances']) == 1 and not dialog['user']['profile']['name']:
            #     response['hellobot']['confidence'] = 1.
            if 'sberchat' in response:
                response['sberchat'] = 0.85
        self.skill_responses = skill_responses
        return skill_responses
