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

import re
from typing import List, Callable, Optional

from deeppavlov.skills.dsl_skill.context import UserContext
from deeppavlov.skills.dsl_skill.handlers.handler import Handler


class RegexHandler(Handler):
    """
    This handler checks whether the message that is passed to it is matched by a regex.

    Adds the following key to ```context.handler_payload```:
        - 'regex_groups' - groups parsed from regular expression in command, by name

    Attributes:
        func: handler function
        state: state in which handler can be activated
        priority: priority of the function. If 2 or more handlers can be activated, function
         with the highest priority is selected
        context_condition: predicate that accepts user context and checks if the handler should be activated.
         Example: `lambda context: context.user_id != 1` checks if user_id is not equal to 1.
         That means a user with id 1 will be always ignored by the handler.
        commands: handler is activated if regular expression from this list is matched with a user message

    """

    def __init__(self,
                 func: Callable,
                 commands: Optional[List[str]] = None,
                 state: Optional[str] = None,
                 context_condition: Optional[Callable] = None,
                 priority: int = 0):
        super().__init__(func, state, context_condition, priority)
        self.commands = [re.compile(command) for command in commands]

    def check(self, context: UserContext) -> bool:
        """
        Checks:
         - if the handler function should be triggered based on the given context via context condition.
         - if at least one of the commands is matched to the `context.message`.

        Args:
            context: user context

        Returns:
            True, if handler should be activated, False otherwise
        """
        is_previous_matches = super().check(context)
        if not is_previous_matches:
            return False

        message = context.message
        return any(re.search(regexp, ' '.join(message)) for regexp in self.commands)

    def expand_context(self, context: UserContext) -> UserContext:
        context.handler_payload = {'regex_groups': {}}
        message = context.message
        for regexp in self.commands:
            match = re.search(regexp, ' '.join(message))
            if match is not None:
                for group_ind, span in enumerate(match.regs):
                    context.handler_payload['regex_groups'][group_ind] = message[span[0]: span[1]]
                for group_name, group_ind in regexp.groupindex.items():
                    context.handler_payload['regex_groups'][group_name] = \
                        context.handler_payload['regex_groups'][group_ind]
                return context
