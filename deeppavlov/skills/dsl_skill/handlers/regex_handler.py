import re
from typing import List, Pattern, Callable

from .handler import Handler


class RegexHandler(Handler):
    """
    This handler checks whether the message that is passed to it is matched by a regex.

    Adds the following field to `context`:
        - context.regex_groups - groups parsed from regular expression in command, by name
    """

    def __init__(self,
                 func: Callable,
                 commands: List[Pattern],
                 state: str = None,
                 context_condition=None,
                 priority: int = 0):
        super().__init__(func, state, context_condition, priority)
        self.commands = [re.compile(command) for command in commands]

    def check(self, message: str = None, context=None):
        is_previous_matches, previous_context = super().check(message, context)
        if not is_previous_matches:
            return False, previous_context

        for regexp in self.commands:
            match = re.search(regexp, ' '.join(message))
            if match is not None:
                regex_groups = dict()
                for group_ind, span in enumerate(match.regs):
                    regex_groups[group_ind] = message[span[0]: span[1]]
                for group_name, group_ind in regexp.groupindex.items():
                    regex_groups[group_name] = regex_groups[group_ind]
                return True
        return False
