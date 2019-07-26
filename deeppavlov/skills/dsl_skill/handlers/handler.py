# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    def expand_context(self, context: UserContext) -> UserContext:
        context.handler_payload = {}
        return context
