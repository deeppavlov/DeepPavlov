import numpy as np

from deeppavlov.models.go_bot.nlg_mechanism import NLGHandler
from deeppavlov.models.go_bot.nlu_mechanism import NLUHandler
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
    def from_configured(nlg_handler: NLGHandler, nlu_handler: NLUHandler, tracker: FeaturizedTracker):
        return FeaturesParams(nlg_handler.num_of_known_actions(),
                              nlu_handler.num_of_known_intents(),
                              tracker.num_features)
