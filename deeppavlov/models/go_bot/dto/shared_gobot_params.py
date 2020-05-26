from deeppavlov.models.go_bot.nlu.nlu_manager import NLUManagerInterface
from deeppavlov.models.go_bot.nlg.nlg_manager import NLGManagerInterface
from deeppavlov.models.go_bot.tracker.featurized_tracker import FeaturizedTracker


# todo logging
class SharedGoBotParams:
    """the DTO-like class to share the params used in various parts of the GO-bot pipeline."""
    # possibly useful: seems like the params reflect only "real-world" knowledge.
    num_actions: int
    num_intents: int
    num_tracker_features: int

    def __init__(self, num_actions: int, num_intents: int, num_tracker_features: int):
        self.num_actions = num_actions
        self.num_intents = num_intents
        self.num_tracker_features = num_tracker_features

    @staticmethod
    def from_configured(nlg_manager: NLGManagerInterface, nlu_manager: NLUManagerInterface, tracker: FeaturizedTracker):
        """builds the params object given some GO-bot units that are already configured"""
        return SharedGoBotParams(nlg_manager.num_of_known_actions(),
                                 nlu_manager.num_of_known_intents(),
                                 tracker.num_features)
