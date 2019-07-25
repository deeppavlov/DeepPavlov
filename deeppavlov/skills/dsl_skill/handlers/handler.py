from typing import Callable, Optional

from deeppavlov.skills.dsl_skill.context import UserContext
from deeppavlov.skills.dsl_skill.utils import SkillResponse


class Handler:
    """
    _Handler instance helps DSLMeta class distinguish functions wrapped
    by @DSLMeta.handler to add them to handlers storage.
    """

    def __init__(self,
                 func: Callable,
                 state: Optional[str] = None,
                 context_condition: Optional[Callable] = None,
                 priority: int = 0):
        self.func = func
        self.state = state
        self.context_condition = context_condition
        self.priority = priority

    def __call__(self, context: UserContext) -> SkillResponse:
        return self.func(context)

    def check(self, context: UserContext) -> bool:
        if self.context_condition is not None:
            return self.context_condition(context)
        return True

    def expand_context(self, context: UserContext):
        context.handler_payload = {}
