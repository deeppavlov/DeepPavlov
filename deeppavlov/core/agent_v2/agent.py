from typing import Sequence, Hashable

from deeppavlov.core.agent_v2.preprocessor import Preprocessor
from deeppavlov.core.agent_v2.states_manager import StatesManager

TG_START_UTT = '\\start'


class Agent:
    def __init__(self, states_manager: StatesManager, preprocessor: Preprocessor) -> None:
        self.states_manager = states_manager
        self.preprocessor = preprocessor

    def __call__(self, utterances: Sequence[str], user_telegram_ids: Sequence[Hashable]):
        should_reset = [utterance == TG_START_UTT for utterance in utterances]
        dialog_states = self.states_manager.get_user_states(user_telegram_ids, should_reset)
        annotations = self.preprocessor(utterances)
        ...
