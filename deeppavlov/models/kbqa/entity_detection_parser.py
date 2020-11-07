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

from typing import List, Tuple, Union, Dict
from collections import defaultdict

import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('question_sign_checker')
class QuestionSignChecker(Component):
    """This class adds question sign if it is absent or replaces dot with question sign"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, questions: List[str]) -> List[str]:
        questions_sanitized = []
        for question in questions:
            if not question.endswith('?'):
                if question.endswith('.'):
                    question = question[:-1] + '?'
                else:
                    question += '?'
            questions_sanitized.append(question)
        return questions_sanitized


@register('entity_detection_parser')
class EntityDetectionParser(Component):
    """This class parses probabilities of tokens to be a token from the entity substring."""

    def __init__(self, entity_tags: List[str], type_tag: str, o_tag: str, tags_file: str, ignore_points: bool = False,
                 return_entities_with_tags: bool = False, thres_proba: float = 0.8, **kwargs):
        """

        Args:
            entity_tags: tags for entities
            type_tag: tag for types
            o_tag: tag for tokens which are neither entities nor types
            tags_file: filename with NER tags
            ignore_points: whether to consider points as separate symbols
            return_entities_with_tags: whether to return a dict of tags (keys) and list of entity substrings (values)
                or simply a list of entity substrings
            thres_proba: if the probability of the tag is less than thres_proba, we assign the tag as 'O'
        """
        self.entity_tags = entity_tags
        self.type_tag = type_tag
        self.o_tag = o_tag
        self.ignore_points = ignore_points
        self.return_entities_with_tags = return_entities_with_tags
        self.thres_proba = thres_proba
        self.tag_ind_dict = {}
        with open(str(expand_path(tags_file))) as fl:
            tags = [line.split('\t')[0] for line in fl.readlines()]
            self.entity_prob_ind = {entity_tag: [i for i, tag in enumerate(tags) if entity_tag in tag]
                                    for entity_tag in self.entity_tags}
            self.type_prob_ind = [i for i, tag in enumerate(tags) if self.type_tag in tag]
            self.et_prob_ind = [i for tag, ind in self.entity_prob_ind.items() for i in ind] + self.type_prob_ind
            for entity_tag, tag_ind in self.entity_prob_ind.items():
                for ind in tag_ind:
                    self.tag_ind_dict[ind] = entity_tag
            for ind in self.type_prob_ind:
                self.tag_ind_dict[ind] = self.type_tag
            self.tag_ind_dict[0] = self.o_tag

    def __call__(self, question_tokens: List[List[str]], token_probas: List[List[List[float]]]) -> \
            Tuple[List[Union[List[str], Dict[str, List[str]]]], List[List[str]],
                  List[Union[List[int], Dict[str, List[List[int]]]]]]:
        """

        Args:
            question_tokens: tokenized questions
            token_probas: list of probabilities of question tokens
        Returns:
            Batch of dicts where keys are tags and values are substrings corresponding to tags
            Batch of substrings which correspond to entity types
            Batch of lists of token indices in the text which correspond to entities
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
        """
        This method makes a list of tags from a list of probas for tags

        Args:
            probas: probabilities for tokens to belong to particular tags

        Returns:
            list of tags for tokens
            list of probabilities of these tags
        """
        tags = []
        tag_probas = []
        for proba in probas:
            tag_num = np.argmax(proba)
            if tag_num in self.et_prob_ind:
                if proba[tag_num] < self.thres_proba:
                    tag_num = 0
            else:
                tag_num = 0
            tags.append(self.tag_ind_dict[tag_num])
            tag_probas.append(proba[tag_num])

        return tags, tag_probas

    def entities_from_tags(self, tokens, tags, tag_probas):
        """
        This method makes lists of substrings corresponding to entities and entity types
        and a list of indices of tokens which correspond to entities

        Args:
            tokens: list of tokens of the text
            tags: list of tags for tokens
            tag_probas: list of probabilities of tags

        Returns:
            list of entity substrings (or a dict of tags (keys) and entity substrings (values))
            list of substrings for entity types
            list of indices of tokens which correspond to entities (or a dict of tags (keys)
                and list of indices of entity tokens)
        """
        entities_dict = defaultdict(list)
        entity_types = []
        entity_dict = defaultdict(list)
        entity_positions_dict = defaultdict(list)
        entities_positions_dict = defaultdict(list)
        entity_type = []
        types_probas = []
        type_proba = []
        replace_tokens = [(' - ', '-'), ("'s", ''), (' .', ''), ('{', ''), ('}', ''),
                          ('  ', ' '), ('"', "'"), ('(', ''), (')', '')]

        cnt = 0
        for n, (tok, tag, proba) in enumerate(zip(tokens, tags, tag_probas)):
            if tag in self.entity_tags:
                if self.ignore_points:
                    if len(tok) == 1 and n < len(tokens) - 1 and tokens[n + 1] == ".":
                        entity_dict[tag].append(f"{tok}.")
                    else:
                        entity_dict[tag].append(tok)
                else:
                    entity_dict[tag].append(tok)
                entity_positions_dict[tag].append(cnt)

            elif tag == self.type_tag:
                entity_type.append(tok)
                type_proba.append(proba)
            elif self.ignore_points and tok == "." and n > 0 and len(tokens[n - 1]) == 1:
                cnt -= 1
            elif any(entity_dict.values()):
                for tag, entity in entity_dict.items():
                    entity = ' '.join(entity)
                    for old, new in replace_tokens:
                        entity = entity.replace(old, new)
                    if entity:
                        entities_dict[tag].append(entity)
                        entities_positions_dict[tag].append(entity_positions_dict[tag])
                    entity_dict[tag] = []
                    entity_positions_dict[tag] = []
            elif len(entity_type) > 0:
                entity_type = ' '.join(entity_type)
                for old, new in replace_tokens:
                    entity_type = entity_type.replace(old, new)
                entity_types.append(entity_type)
                entity_type = []
                types_probas.append(np.mean(type_proba))
                type_proba = []
            cnt += 1

        if entity_types:
            entity_types = sorted(zip(entity_types, types_probas), key=lambda x: x[1], reverse=True)
            entity_types = [entity_type[0] for entity_type in entity_types]

        entities_list = [entity for tag, entities in entities_dict.items() for entity in entities]
        entities_positions_list = [position for tag, positions in entities_positions_dict.items()
                                   for position in positions]

        if self.return_entities_with_tags:
            return entities_dict, entity_types, entities_positions_dict
        else:
            return entities_list, entity_types, entities_positions_list
