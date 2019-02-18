from datetime import datetime
from typing import Sequence, Hashable, Any
from itertools import compress
import operator

from deeppavlov.core.agent_v2.preprocessor import Preprocessor
from deeppavlov.core.agent_v2.states_manager import StatesManager, TG_START_UTT


class Agent:
    def __init__(self, states_manager: StatesManager, preprocessor: Preprocessor) -> None:
        self.states_manager = states_manager
        self.preprocessor = preprocessor

    def __call__(self, utterances: Sequence[str], user_telegram_ids: Sequence[Hashable],
                 user_device_types: Sequence[Any],
                 date_times: Sequence[datetime], locations=Sequence[Any], channel_types=Sequence[str]):
        should_reset = [utterance == TG_START_UTT for utterance in utterances]
        # here and further me stands for mongoengine
        me_users = self.states_manager.get_users(user_telegram_ids, user_device_types)
        annotations = self.predict_annotations(utterances, should_reset)
        me_utterances = self.states_manager.get_utterances(utterances, annotations, me_users, date_times)
        me_states = self.states_manager.get_states(me_users, me_utterances, locations, channel_types, should_reset)

        # DEBUG
        # for state in me_states:
        #     print(state.to_dict())

    def predict_annotations(self, utterances, should_reset):
        informative_utterances = list(compress(utterances, map(operator.not_, should_reset)))
        annotations = iter(self.preprocessor(informative_utterances))
        for reset in should_reset:
            if reset:
                yield None
            else:
                yield next(annotations)
