from typing import Callable, Optional, List, Tuple

from deeppavlov.skills.dsl_skill.context import UserContext
from deeppavlov.skills.dsl_skill.utils import ResponseType


class Handler:
    """
    _Handler instance helps DSLMeta class distinguish functions wrapped
    by @DSLMeta.handler to add them to handlers storage.
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

    def __call__(self, message: Optional[str] = None, context: Optional[UserContext] = None) -> ResponseType:
        return self.func(message, context)

    def check(self, context: Optional[UserContext] = None) -> bool:
        if self.context_condition is not None:
            return self.context_condition(context)
        return True
