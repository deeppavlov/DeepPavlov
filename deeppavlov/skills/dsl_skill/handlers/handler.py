from typing import Callable, Tuple

from deeppavlov.skills.dsl_skill.utils import ResponseType


class Handler:
    """
    _Handler instance helps ZDialog class distinguish functions wrapped
    by @ZDialog.handler to add them to handlers storage.
    """

    def __init__(self,
                 func: Callable,
                 state: str = None,
                 context_condition=None,
                 priority: int = 0):
        self.func = func
        self.state = state
        self.context_condition = context_condition
        self.priority = priority

    def __call__(self, message: str = None, context=None) -> ResponseType:
        return self.func(message, context)

    def check(self, message: str = None, context=None) -> Tuple[bool, 'UserContext']:
        if self.context_condition is not None:
            return self.context_condition(context), context
        return True, context

