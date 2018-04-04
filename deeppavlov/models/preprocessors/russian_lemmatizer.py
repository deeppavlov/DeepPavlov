from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
import pymorphy2


@register('pymorphy_russian_lemmatizer')
class PymorphyRussianLemmatizer(Component):
    def __init__(self):
        self.lemmatizer = pymorphy2.MorphAnalyzer()

    def __call__(self, tokens_batch, **kwargs):
        """Takes batch of tokens and returns the lemmatized tokens"""
        lemma_batch = []
        for utterance in tokens_batch:
            lemma_utterance = []
            for token in utterance:
                p = self.lemmatizer.parse(token)[0]
                lemma_utterance.append(p.normal_form)
            lemma_batch.append(lemma_utterance)
        return lemma_batch
