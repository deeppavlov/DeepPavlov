from datetime import datetime
from typing import Sequence, Hashable, Any
from itertools import compress
import operator

from deeppavlov.core.agent_v2.preprocessor import Preprocessor
from deeppavlov.core.agent_v2.state_manager import StateManager, TG_START_UTT
from deeppavlov.core.agent_v2.skill_manager import SkillManager


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
        me_users = self.state_manager.get_users(user_telegram_ids, user_device_types)
        annotations = self.predict_annotations(utterances, should_reset)
        me_utterances = self.state_manager.get_utterances(utterances, annotations, me_users, date_times)
        me_dialos = self.state_manager.get_dialogs(me_users, me_utterances, locations, channel_types, should_reset)
        state = self.state_manager.get_state(me_dialos)
        responses = self.skill_manager(state)

        # TODO
        # After response is chosen for each dialog in the state, dialog objects should be updated and saved to DB.
        # Then response utterances should be sent to the users.

    def predict_annotations(self, utterances, should_reset):
        informative_utterances = list(compress(utterances, map(operator.not_, should_reset)))
        annotations = iter(self.preprocessor(informative_utterances))
        for reset in should_reset:
            if reset:
                yield None
            else:
                yield next(annotations)
