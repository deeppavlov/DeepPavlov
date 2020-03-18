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

from typing import List
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('entity_detection_parser')
class EntityDetectionParser(Component):
    """
        This class parses probabilities of tokens to be a token from the entity substring
    """

    def __init__(self, thres_proba: float = 0.86, **kwargs):
        self.thres_proba = thres_proba
        pass

    def __call__(self, question_tokens: List[List[str]],
                 token_probas: List[List[List[float]]], **kwargs):
        tokens, probas = question_tokens[0], token_probas[0]

        tags = []
        for proba in probas:
            if proba[0] <= self.thres_proba:
                tags.append(1)
            if proba[0] > self.thres_proba:
                tags.append(0)

        entities = self.entities_from_tags(tokens, tags, probas)
        return entities

    def entities_from_tags(self, tokens, tags, probas):
        entities = []
        start = 0
        entity = ''
        replace_tokens = [(' - ', '-'), ("'s", ''), (' .', ''), ('{', ''), ('}', '')]

        for tok, tag, proba in zip(tokens, tags, probas):
            if tag != 0 and start == 0:
                start = 1
                entity = tok
            elif tag != 0 and start == 1:
                entity += ' '
                entity += tok
            elif tag == 0 and len(entity) > 0 and start == 1:
                start = 0
                for replace_token in replace_tokens:
                    entity = entity.replace(replace_token[0], replace_token[1])
                entities.append(entity)
            else:
                pass

        return entities
