from abc import ABCMeta, abstractmethod
from typing import List

from deeppavlov.models.go_bot.dto.dataset_features import BatchDialoguesFeatures
from deeppavlov.models.go_bot.nlg.dto.nlg_response_interface import NLGResponseInterface
from deeppavlov.models.go_bot.policy.dto.policy_prediction import PolicyPrediction


class NLGManagerInterface(metaclass=ABCMeta):

    @abstractmethod
    def get_action_id(self, action_text) -> int:
        """
        Looks up for an ID relevant to the passed action text in the list of known actions and their ids.

        Args:
            action_text: the text for which an ID needs to be returned.
        Returns:
            an ID corresponding to the passed action text
        """
        pass

    @abstractmethod
    def get_api_call_action_id(self) -> int:
        """
        Returns:
            an ID corresponding to the api call action
        """
        pass

    @abstractmethod
    def decode_response(self,
                        utterance_batch_features: BatchDialoguesFeatures,
                        policy_prediction: PolicyPrediction,
                        tracker_slotfilled_state) -> NLGResponseInterface:
        # todo: docstring
        pass

    @abstractmethod
    def num_of_known_actions(self) -> int:
        """
        Returns:
            the number of actions known to the NLG module
        """
        pass

    @abstractmethod
    def known_actions(self) -> List:
        """
        Returns:
             the list of actions known to the NLG module
        """
