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

import pymorphy2

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('pymorphy_russian_lemmatizer')
class PymorphyRussianLemmatizer(Component):
    """Class for lemmatization using PyMorphy."""

    def __init__(self, *args, **kwargs):
        self.lemmatizer = pymorphy2.MorphAnalyzer()

    def __call__(self, tokens_batch, **kwargs):
        """Takes batch of tokens and returns the lemmatized tokens."""
        lemma_batch = []
        for utterance in tokens_batch:
            lemma_utterance = []
            for token in utterance:
                p = self.lemmatizer.parse(token)[0]
                lemma_utterance.append(p.normal_form)
            lemma_batch.append(lemma_utterance)
        return lemma_batch
