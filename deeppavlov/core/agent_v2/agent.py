from typing import Collection, Hashable

from deeppavlov.core.agent_v2.preprocessor import Preprocessor


class Agent:
    def __init__(self, states_manager, preprocessor: Preprocessor) -> None:
        self.states_manager = states_manager
        self.preprocessor = preprocessor

    def __call__(self, utterances: Collection[str], user_ids: Collection[Hashable]):
        should_reset = [utterance == '\\start' for utterance in utterances]
        dialog_states = self.states_manager.get(user_ids, should_reset)
        annotations = self.preprocessor(utterances)
        ...
