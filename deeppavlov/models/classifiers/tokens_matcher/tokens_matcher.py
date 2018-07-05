from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('tokens_matcher')
class TokensMatcher(Component):
    def __init__(self, words, *args, **kwargs):
        self.words = set(words)

    def __call__(self, tokens_batch):
        return [float(any(word in ' '.join(tokens) for word in self.words)) for tokens in tokens_batch]
