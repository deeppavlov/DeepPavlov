# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

import random
import re
from typing import List, Tuple, Optional

from deeppavlov.deprecated.skill import Skill


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
        default_confidence: The default confidence.

    Attributes:
        responses: List of str responses from which response will be randomly
            selected.
        patterns: List of str patterns for utterance matching. Patterns may
            be all plain texts or all regexps.
        regex: Turns on regular expressions matching mode.
        ignore_case: Turns on utterances case ignoring.
        default_confidence: The default confidence.
    """

    def __init__(self, responses: List[str], patterns: Optional[List[str]] = None,
                 regex: bool = False, ignore_case: bool = True, default_confidence: float = 1) -> None:
        if isinstance(responses, str):
            responses = [responses]
        self.responses = responses
        if isinstance(patterns, str):
            patterns = [patterns]
        self.regex = regex
        self.ignore_case = ignore_case
        self.default_confidence = default_confidence
        if regex:
            if patterns:
                flags = re.IGNORECASE if ignore_case else 0
                patterns = [re.compile(pattern, flags) for pattern in patterns]
        else:
            if patterns and ignore_case:
                patterns = [pattern.lower() for pattern in patterns]
        self.patterns = patterns

    def __call__(self, utterances_batch: list, history_batch: list,
                 states_batch: Optional[list] = None) -> Tuple[list, list]:
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
            confidence = [self.default_confidence] * len(utterances_batch)
        else:
            if self.ignore_case:
                utterances_batch = [utterance.lower() for utterance in utterances_batch]
            if self.regex:
                confidence = [
                    self.default_confidence * float(any([pattern.search(utterance) for pattern in self.patterns]))
                    for utterance in utterances_batch]
            else:
                confidence = [self.default_confidence * float(any([pattern in utterance for pattern in self.patterns]))
                              for utterance in utterances_batch]

        return response, confidence
