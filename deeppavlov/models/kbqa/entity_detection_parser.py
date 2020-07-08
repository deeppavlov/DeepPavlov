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

from typing import List, Tuple

import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('entity_detection_parser')
class EntityDetectionParser(Component):
    """This class parses probabilities of tokens to be a token from the entity substring."""

    def __init__(self, entity_tag: str, type_tag: str, o_tag: str, tags_file: str, thres_proba: float = 0.8, **kwargs):
        """

        Args:
            entity_tag: tag for entities
            type_tag: tag for types
            o_tag: tag for tokens which are neither entities nor types
            tags_file: filename with NER tags
            thres_proba: if the probability of the tag is less than thres_proba, we assign the tag as 'O'
        """
        self.entity_tag = entity_tag
        self.type_tag = type_tag
        self.o_tag = o_tag
        self.thres_proba = thres_proba
        self.tag_ind_dict = {}
        with open(str(expand_path(tags_file))) as fl:
            tags = [line.split('\t')[0] for line in fl.readlines()]
            self.entity_prob_ind = [i for i, tag in enumerate(tags) if self.entity_tag in tag]
            self.type_prob_ind = [i for i, tag in enumerate(tags) if self.type_tag in tag]
            self.et_prob_ind = self.entity_prob_ind + self.type_prob_ind
            for ind in self.entity_prob_ind:
                self.tag_ind_dict[ind] = self.entity_tag
            for ind in self.type_prob_ind:
                self.tag_ind_dict[ind] = self.type_tag
            self.tag_ind_dict[0] = self.o_tag

    def __call__(self, question_tokens: List[List[str]],
                 token_probas: List[List[List[float]]]) -> Tuple[List[List[str]], List[List[str]], List[List[List[int]]]]:
        """

        Args:
            question_tokens: tokenized questions
            token_probas: list of probabilities of question tokens
        """
        entities_batch = []
        types_batch = []
        positions_batch = []
        for tokens, probas in zip(question_tokens, token_probas):
            tags, tag_probas = self.tags_from_probas(probas)
            entities, types, positions = self.entities_from_tags(tokens, tags, tag_probas)
            entities_batch.append(entities)
            types_batch.append(types)
            positions_batch.append(positions)
        return entities_batch, types_batch, positions_batch

    def tags_from_probas(self, probas):
        tag_list = [self.o_tag, self.entity_tag, self.type_tag]
        tags = []
        tag_probas = []
        for proba in probas:
            tag_num = np.argmax(proba)
            if tag_num in self.et_prob_ind:
                if proba[tag_num] < self.thres_proba:
                    tag_num = 0
            tags.append(self.tag_ind_dict[tag_num])
            tag_probas.append(proba[tag_num])
                    
        return tags, tag_probas

    def entities_from_tags(self, tokens, tags, tag_probas):
        entities = []
        entity_types = []
        entity = []
        entity_positions = []
        entities_positions = []
        entity_type = []
        types_probas = []
        type_proba = []
        replace_tokens = [(' - ', '-'), ("'s", ''), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'"), ('(', ''), (')', '')]

        for n, (tok, tag, proba) in enumerate(zip(tokens, tags, tag_probas)):
            if tag == self.entity_tag:
                entity.append(tok)
                entity_positions.append(n)
            elif tag == self.type_tag:
                entity_type.append(tok)
                type_proba.append(proba)
            elif len(entity) > 0:
                entity = ' '.join(entity)
                for old, new in replace_tokens:
                    entity = entity.replace(old, new)
                entities.append(entity)
                entities_positions.append(entity_positions)
                entity = []
                entity_positions = []
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

        return entities, entity_types, entities_positions
