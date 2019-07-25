import re
from typing import List, Callable, Optional

from deeppavlov.skills.dsl_skill.context import UserContext
from .handler import Handler


class RegexHandler(Handler):
    """
    This handler checks whether the message that is passed to it is matched by a regex.

    Adds the following field to `context`:
        - context.regex_groups - groups parsed from regular expression in command, by name
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
        is_previous_matches = super().check(context)
        if not is_previous_matches:
            return False

        message = context.message
        return any(re.search(regexp, ' '.join(message)) for regexp in self.commands)

    def expand_context(self, context: UserContext):
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
                return
