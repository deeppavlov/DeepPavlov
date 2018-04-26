import unicodedata
import sys
import re

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register('sanitizer')
class Sanitizer(Component):
    def __init__(self, diacritical=True, nums=False, *args, **kwargs):
        """Remove all combining characters like diacritical marks from tokens"""
        self.diacritical = diacritical
        self.nums = nums
        self.combining_characters = dict.fromkeys([c for c in range(sys.maxunicode)
                                                   if unicodedata.combining(chr(c))])

    def filter_diacritical(self, tokens_batch):
        """Takes batch of tokens and returns the batch with sanitized tokens"""
        sanitized_batch = []
        for utterance in tokens_batch:
            sanitized_utterance = []
            for token in utterance:
                token = unicodedata.normalize('NFD', token)
                sanitized_utterance.append(token.translate(self.combining_characters))
            sanitized_batch.append(sanitized_utterance)
        return sanitized_batch

    def replace_nums(self, tokens_batch):
        sanitized_batch = []
        for utterance in tokens_batch:
            sanitized_batch.append([re.sub('[0-9]', '1', token) for token in utterance])
        return sanitized_batch

    def __call__(self, tokens_batch, **kwargs):
        if self.filter_diacritical:
            tokens_batch = self.filter_diacritical(tokens_batch)
        if self.nums:
            tokens_batch = self.replace_nums(tokens_batch)
        return tokens_batch
