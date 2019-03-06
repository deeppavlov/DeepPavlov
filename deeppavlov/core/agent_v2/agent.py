from datetime import datetime
from typing import Sequence, Hashable, Any
from itertools import compress
import operator

from deeppavlov.core.agent_v2.rest_caller import RestCaller
from deeppavlov.core.agent_v2.state_manager import StateManager
from deeppavlov.core.agent_v2.skill_manager import SkillManager
from deeppavlov.core.agent_v2.hardcode_utterances import TG_START_UTT


class Agent:
    def __init__(self, state_manager: StateManager, preprocessor: RestCaller,
                 skill_manager: SkillManager) -> None:
        self.state_manager = state_manager
        self.preprocessor = preprocessor
        self.skill_manager = skill_manager

    def __call__(self, utterances: Sequence[str], user_telegram_ids: Sequence[Hashable],
                 user_device_types: Sequence[Any],
                 date_times: Sequence[datetime], locations=Sequence[Any], channel_types=Sequence[str]):
        should_reset = [utterance == TG_START_UTT for utterance in utterances]
        # here and further me stands for mongoengine
        me_users = self.state_manager.get_users(user_telegram_ids, user_device_types)
        me_utterances = self.state_manager.get_utterances(utterances, me_users, date_times)
        me_dialogs = self.state_manager.get_dialogs(me_users, me_utterances, locations, channel_types, should_reset)
        informative_dialogs = list(compress(me_dialogs, map(operator.not_, should_reset)))

        annotations = self.preprocessor(self.state_manager.get_state(informative_dialogs), should_reset)
        for utt, ann in zip(me_utterances, annotations):
            utt.annotations = ann
            utt.save()

        state = self.state_manager.get_state(me_dialogs)

        skill_names, utterances, confidences = self.skill_manager(state)

        self.state_manager.add_bot_utterances(me_dialogs, utterances, [datetime.utcnow()] * len(me_dialogs),
                                              skill_names, confidences)

        return utterances  # return text only to the users
