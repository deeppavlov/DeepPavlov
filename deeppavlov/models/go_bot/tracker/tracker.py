from abc import ABCMeta, abstractmethod
from typing import Union, List, Tuple, Any, Dict

import numpy as np


class Tracker(metaclass=ABCMeta):
    """
    An abstract class for trackers: a model that holds a dialogue state and
    generates state features.
    """

    @abstractmethod
    def update_state(self, slots: Union[List[Tuple[str, Any]], Dict[str, Any]]) -> None:
        """
        Updates dialogue state with new ``slots``, calculates features.

        Returns:
            Tracker: ."""
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