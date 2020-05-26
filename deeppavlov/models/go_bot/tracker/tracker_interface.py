from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import numpy as np

from deeppavlov.models.go_bot.nlu.dto.nlu_response_interface import NLUResponseInterface
from deeppavlov.models.go_bot.tracker.dto.tracker_knowledge_interface import TrackerKnowledgeInterface


class TrackerInterface(metaclass=ABCMeta):
    """
    An abstract class for trackers: a model that holds a dialogue state and
    generates state features.
    """

    @abstractmethod
    def update_state(self, nlu_response: NLUResponseInterface) -> None:
        """Updates dialogue state with new ``slots``, calculates features."""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: dictionary with current slots and their values."""
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Resets dialogue state"""
        pass

    @abstractmethod
    def get_features(self) -> np.ndarray:
        """
        Returns:
            np.ndarray[float]: numpy array with calculates state features."""
        pass

    @abstractmethod
    def get_current_knowledge(self) -> TrackerKnowledgeInterface:
        pass
