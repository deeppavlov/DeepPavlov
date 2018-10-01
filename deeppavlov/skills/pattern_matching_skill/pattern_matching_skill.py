import random
import re

from deeppavlov.core.models.component import Component


class PatternMatchingSkill(Component):
    """
    Create skills as pre-defined responses for a user's input containing specific keywords or regular expressions.
    Every skill returns response and confidence.

    Examples:

        >>> # Import key components to build HelloBot.
        >>> from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill
        >>> from deeppavlov.core.agent import Agent, HighestConfidenceSelector
        >>> hello = PatternMatchingSkill(responses=['Hello world!'], patterns=["hi", "hello", "good day"])
        >>> bye = PatternMatchingSkill(['Goodbye world!', 'See you around'], patterns=["bye", "chao", "see you"])
        >>> fallback = PatternMatchingSkill(["I don't understand, sorry", 'I can say "Hello world!"'])
        >>> # Agent executes skills and then takes response from the skill with the highest confidence.
        >>> agent = Agent([hello, bye, fallback], skills_selector=HighestConfidenceSelector())
        >>> agent(['Hello', 'Bye', 'Or not'])
        ['Hello world!', 'Goodbye world!', 'I can say "Hello world!"']

    """
    def __init__(self, responses, patterns=None, regex=False, ignore_case=True):
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
