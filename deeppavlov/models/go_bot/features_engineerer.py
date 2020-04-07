import numpy as np

from deeppavlov.models.go_bot.nlu_manager import NLUManager
from deeppavlov.models.go_bot.nlg_manager import NLGManager
from deeppavlov.models.go_bot.tracker.featurized_tracker import FeaturizedTracker


class FeaturesParams:

    num_actions: int
    num_intents: int
    num_tracker_features: int

    def __init__(self, num_actions, num_intents, num_tracker_features):
        self.num_actions = num_actions
        self.num_intents = num_intents
        self.num_tracker_features = num_tracker_features

    @staticmethod
    def from_configured(nlg_manager: NLGManager, nlu_manager: NLUManager, tracker: FeaturizedTracker):
        return FeaturesParams(nlg_manager.num_of_known_actions(),
                              nlu_manager.num_of_known_intents(),
                              tracker.num_features)
