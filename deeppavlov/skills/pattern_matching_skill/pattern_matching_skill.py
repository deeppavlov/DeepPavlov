import random
import re
from typing import List, Tuple, Optional

from deeppavlov.core.skill.skill import Skill


class PatternMatchingSkill(Skill):
    """Skill, matches utterances to patterns, returns predefined answers.

    Allows to create skills as pre-defined responses for a user's input
    containing specific keywords or regular expressions. Every skill returns
    response and confidence.

    Args:
        responses: List of str responses from which response will be randomly
            selected.
        patterns: List of str patterns for utterance matching. Patterns may
            be all plain texts or all regexps.
        regex: Turns on regular expressions matching mode.
        ignore_case: Turns on utterances case ignoring.

    Attributes:
        responses: List of str responses from which response will be randomly
            selected.
        patterns: List of str patterns for utterance matching. Patterns may
            be all plain texts or all regexps.
        regex: Turns on regular expressions matching mode.
        ignore_case: Turns on utterances case ignoring.
    """
    def __init__(self, responses: List[str], patterns: Optional[List[str]]=None,
                 regex: bool=False, ignore_case: bool=True) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self.responses = responses
        if isinstance(patterns, str):
            patterns = [patterns]
        self.regex = regex
        self.ignore_case = ignore_case
        if regex:
            if patterns:
                flags = re.IGNORECASE if ignore_case else 0
                patterns = [re.compile(pattern, flags) for pattern in patterns]
        else:
            if patterns and ignore_case:
                patterns = [pattern.lower() for pattern in patterns]
        self.patterns = patterns

    def __call__(self, utterances_batch: list, history_batch: list,
                 states_batch: Optional[list]=None) -> Tuple[list, list]:
        """Returns skill inference result.

        Returns batches of skill inference results, estimated confidence
        levels and up to date states corresponding to incoming utterance
        batch.

        Args:
            utterances_batch: A batch of utterances of any type.
            history_batch: A batch of list typed histories for each utterance.
            states_batch: Optional. A batch of arbitrary typed states for
                each utterance.

        Returns:
            response: A batch of arbitrary typed skill inference results.
            confidence: A batch of float typed confidence levels for each of
                skill inference result.
        """
        response = [random.choice(self.responses) for _ in utterances_batch]
        if self.patterns is None:
            confidence = [0.5] * len(utterances_batch)
        else:
            if self.ignore_case:
                utterances_batch = [utterance.lower() for utterance in utterances_batch]
            if self.regex:
                confidence = [float(any([pattern.search(utterance) for pattern in self.patterns]))
                              for utterance in utterances_batch]
            else:
                confidence = [float(any([pattern in utterance for pattern in self.patterns]))
                              for utterance in utterances_batch]

        return response, confidence
