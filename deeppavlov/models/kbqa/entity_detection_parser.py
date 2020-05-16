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
    """This class parses probabilities of tokens to be a token from the entity substring."""

    def __init__(self, thres_proba: float = 0.86, **kwargs):
        self.thres_proba = thres_proba

    def __call__(self, question_tokens: List[List[str]],
                 token_tags: List[List[List[float]]]) -> List[List[str]]:
        """

        Args:
            question_tokens: tokenized questions
            token_probas: list of probabilities of question tokens to belong to
            "B-TAG" (beginning of entity substring), "I-TAG" (inner token of entity substring)
            or "O-TAG" (not an entity token)
        """
        
        entities_batch = []
        types_batch = []
        for tokens, tags in zip(question_tokens, token_tags):
            entities, types = self.entities_from_tags(tokens, tags)
            entities_batch.append(entities)
            types_batch.append(types)
        return entities_batch, types_batch

    def entities_from_tags(self, tokens, tags):
        entities = []
        entity_types = []
        entity = []
        entity_type = []
        replace_tokens = [(' - ', '-'), ("'s", ''), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''), (')', '')]

        for tok, tag in zip(tokens, tags):
            if tag == "E-TAG":
                entity.append(tok)
            elif tag == "T-TAG":
                entity_type.append(tok)
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

        return entities, entity_types
