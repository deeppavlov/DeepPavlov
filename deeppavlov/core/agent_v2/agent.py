from datetime import datetime
from typing import Sequence, Hashable, Any, Dict
from itertools import compress
import operator

from deeppavlov.core.agent_v2.preprocessor import Preprocessor
from deeppavlov.core.agent_v2.state_manager import StateManager
from deeppavlov.core.agent_v2.skill_manager import SkillManager
from deeppavlov.core.agent_v2.hardcode_utterances import TG_START_UTT
from deeppavlov.core.agent_v2.state_schema import Dialog


class Agent:
    def __init__(self, state_manager: StateManager, preprocessor: Preprocessor,
                 skill_manager: SkillManager) -> None:
        self.state_manager = state_manager
        self.preprocessor = preprocessor
        self.skill_manager = skill_manager

    def __call__(self, utterances: Sequence[str], user_telegram_ids: Sequence[Hashable],
                 user_device_types: Sequence[Any],
                 date_times: Sequence[datetime], locations=Sequence[Any], channel_types=Sequence[str]):
        should_reset = [utterance == TG_START_UTT for utterance in utterances]
        # here and further me stands for mongoengine
        me_users = self.state_manager.get_or_create_users(user_telegram_ids, user_device_types)
        me_dialogs = self.state_manager.get_or_create_dialogs(me_users, locations, channel_types, should_reset)
        self.state_manager.add_user_utterances(me_dialogs, utterances, date_times)
        informative_dialogs = list(compress(me_dialogs, map(operator.not_, should_reset)))

        self._update_annotations(informative_dialogs)

        state = self.state_manager.get_state(me_dialogs)

        skill_names, utterances, confidences, profiles, non_active_skills = self.skill_manager(state)

        self._update_profiles(me_users, profiles)

        self.state_manager.add_bot_utterances(me_dialogs, utterances, [datetime.utcnow()] * len(me_dialogs),
                                              skill_names, confidences, non_active_skills=non_active_skills)

        self._update_annotations(me_dialogs)

        return utterances  # return text only to the users

    def _update_annotations(self, me_dialogs: Sequence[Dialog]) -> None:
        annotations = self.preprocessor(self.state_manager.get_state(me_dialogs))
        utterances = [dialog.utterances[-1] for dialog in me_dialogs]
        self.state_manager.add_annotations(utterances, annotations)

    def _update_profiles(self, me_users, profiles) -> None:
        if not profiles:
            return
        for me_user, profile in zip(me_users, profiles):
            self.state_manager.update_user_profile(me_user, profile)
