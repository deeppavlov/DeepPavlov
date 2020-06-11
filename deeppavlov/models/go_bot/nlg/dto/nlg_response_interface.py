from abc import ABCMeta
from typing import Tuple


class NLGObjectResponseInterface(metaclass=ABCMeta):
    def to_serializable_dict(self) -> dict:
        raise NotImplementedError(f"to_serializable_dict() not implemented in {self.__class__.__name__}")


NLGResponseInterface = Tuple[NLGObjectResponseInterface, str]
