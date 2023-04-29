# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

from typing import List

import spacy

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('spacy_lemmatizer')
class SpacyLemmatizer(Component):
    def __init__(self, model: str, **kwargs):
        self.nlp = spacy.load(model)

    def __call__(self, words_batch: List[List[str]]):
        return [[self.nlp(word)[0].lemma_ for word in words_list] for words_list in words_batch]
