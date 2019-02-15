from datetime import datetime
from typing import Sequence, Hashable, Any
from itertools import compress
from operator import itemgetter

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
        me_states = self.states_manager.get_states(me_users, locations, channel_types, should_reset)
        informative_utterances = list(compress(enumerate(utterances), should_reset))
        annotations = iter(self.preprocessor(informative_utterances))
        all_annotations = [next(annotations) if should_reset[i] else None for i in range(len(should_reset))]
        me_utterances = self.states_manager.get_utterances(utterances, all_annotations, me_users, date_times)

    def predict_annotations(self, utterances):
        return self.preprocessor(utterances)
