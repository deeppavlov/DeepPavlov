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

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('entity_detection_parser')
class EntityDetectionParser(Component):
    """This class parses probabilities of tokens to be a token from the entity substring."""

    def __init__(self, thres_proba: float = 0.8, **kwargs):
        self.thres_proba = thres_proba

    def __call__(self, question_tokens: List[List[str]],
                 token_probas: List[List[List[float]]]) -> List[List[str]]:
        """

        Args:
            question_tokens: tokenized questions
            token_probas: list of probabilities of question tokens to belong to
            "B-TAG" (beginning of entity substring), "I-TAG" (inner token of entity substring)
            or "O-TAG" (not an entity token)
        """
        
        entities_batch = []
        types_batch = []
        for tokens, probas in zip(question_tokens, token_probas):
            tags, tag_probas = self.tags_from_probas(probas)
            entities, types = self.entities_from_tags(tokens, tags, tag_probas)
            entities_batch.append(entities)
            types_batch.append(types)
        return entities_batch, types_batch

    def tags_from_probas(self, probas):
        tag_list = ["O-TAG", "E-TAG", "T-TAG"]
        tags = []
        tag_probas = []
        for proba in probas:
            tag_num = np.argmax(proba)
            if tag_num in [1, 2]:
                if proba[tag_num] < self.thres_proba:
                    tag_num = 0
            tags.append(tag_list[tag_num])
            tag_probas.append(proba[tag_num])
                    
        return tags, tag_probas

    def entities_from_tags(self, tokens, tags, tag_probas):
        entities = []
        entity_types = []
        entity = []
        entity_type = []
        types_probas = []
        type_proba = []
        replace_tokens = [(' - ', '-'), ("'s", ''), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''), (')', '')]

        for tok, tag, proba in zip(tokens, tags, tag_probas):
            if tag == "E-TAG":
                entity.append(tok)
            elif tag == "T-TAG":
                entity_type.append(tok)
                type_proba.append(proba)
            elif len(entity) > 0:
                entity = ' '.join(entity)
                for old, new in replace_tokens:
                    entity = entity.replace(old, new)
                entities.append(entity)
                entity = []
            elif len(entity_type) > 0:
                entity_type = ' '.join(entity_type)
                for old, new in replace_tokens:
                    entity_type = entity_type.replace(old, new)
                entity_types.append(entity_type)
                entity_type = []
                types_probas.append(np.mean(type_proba))
                type_proba = []

        if entity_types:
            entity_types = sorted(zip(entity_types, types_probas), key=lambda x: x[1], reverse=True)
            entity_types = [entity_type[0] for entity_type in entity_types]

        return entities, entity_types
