import random
import re

from deeppavlov.core.models.component import Component


class PatternMatchingSkill(Component):
    def __init__(self, responses, patterns=None, regex=False, ignore_case=True):
        if isinstance(responses, str):
            responses = [responses]
        self.responses = responses
        if isinstance(patterns, str):
            patterns = [patterns]
        if patterns and regex:
            patterns = [re.compile(pattern) for pattern in patterns]
        self.patterns = patterns
        self.regex = regex
        self.ignore_case = ignore_case

    def __call__(self, utterances_batch, history_batch, states_batch):
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
