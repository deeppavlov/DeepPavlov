import unicodedata
import sys

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register('sanitizer')
class Sanitizer(Component):
    def __init__(self, *args, **kwargs):
        """Remove all combining characters like diacritical marks from tokens"""
        self.combining_characters = dict.fromkeys([c for c in range(sys.maxunicode)
                                                   if unicodedata.combining(chr(c))])

    def __call__(self, tokens_batch, **kwargs):
        """Takes batch of tokens and returns the batch with sanitized tokens"""
        sanitized_batch = []
        for utterance in tokens_batch:
            sanitized_utterance = []
            for token in utterance:
                token = unicodedata.normalize('NFD', token)
                sanitized_utterance.append(token.translate(self.combining_characters))
            sanitized_batch.append(sanitized_utterance)
        return sanitized_batch
