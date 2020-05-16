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

import re
from typing import List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import load_pickle


@register('template_matcher')
class TemplateMatcher(Component, Serializable):
    """
        This class matches the question with one of the templates
        to extract entity substrings and define which relations
        corresponds to the question
    """

    def __init__(self, load_path: str, templates_filename: str, return_types: bool = False, **kwargs) -> None:
        """

        Args:
            load_path: path to folder with file with templates
            templates_filename: file with templates
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self._templates_filename = templates_filename
        self.return_types = return_types
        self.load()

    def load(self) -> None:
        self.templates = load_pickle(self.load_path / self._templates_filename)

    def save(self) -> None:
        raise NotImplementedError

    def __call__(self, question: str) -> Tuple[List[str], List[Tuple[str]], str]:
        
        question = question.lower()
        question_length = len(question)
        entities = []
        types = []
        relations = []
        query_type = ""
        min_length = 100
        for template in self.templates:
            template_init = template
            template = "yyy" + template
            if all([not template.endswith(tok) for tok in ["xxx?", "ttt?", "yyy?"]]):
                template = template.replace('?', 'yyy?')
            template_len = len(template.replace('xxx', '').replace('ttt', '').replace('yyy', ''))
            positions = [("xxx", m.start()) for m in re.finditer('xxx', template)] + \
                        [("ttt", m.start()) for m in re.finditer('ttt', template)] + \
                        [("yyy", m.start()) for m in re.finditer('yyy', template)]
            positions = sorted(positions, key=lambda x: x[1])
            positions_entity_tokens = []
            positions_type_tokens = []
            positions_unuseful_tokens = []
            for n, position in enumerate(positions):
                if position[0] == "xxx":
                    positions_entity_tokens.append(n)
                if position[0] == "ttt":
                    positions_type_tokens.append(n)
                if position[0] == "yyy":
                    positions_unuseful_tokens.append(n)
            template_regexp = template
            for template_token in ["xxx", "ttt"]:
                template_regexp = template_regexp.replace(template_token, "([a-zа-я\d\s\.-’,]+)")
            template_regexp = template_regexp.replace("yyy", "([a-zа-я\d\s\.-’,]*)")
            fnd = re.findall(template_regexp, question)
            
            if fnd and str(type(fnd[0])) == "<class 'tuple'>":
                entities_cand = [fnd[0][pos] for pos in positions_entity_tokens]
                types_cand = [fnd[0][pos] for pos in positions_type_tokens]
                unuseful_tokens = [fnd[0][pos] for pos in positions_unuseful_tokens]
                entity_lengths = [len(entity) for entity in entities_cand]
                type_lengths = [len(entity_type) for entity_type in types_cand]
                unuseful_tokens_len = sum([len(unuseful_tok.replace('?', '')) for unuseful_tok in unuseful_tokens])
                print(template_init, template_regexp, fnd)
                print(entities_cand, types_cand, unuseful_tokens, entity_lengths, type_lengths, unuseful_tokens_len, template_len, question_length)

                if 0 not in entity_lengths:
                    cur_len = sum(entity_lengths) + sum(type_lengths)
                    if cur_len < min_length and unuseful_tokens_len + template_len + cur_len == question_length:
                        entities = entities_cand
                        types = types_cand
                        relations = self.templates[template_init][1:]
                        query_type = self.templates[template_init][0]
                        min_length = cur_len

        if self.return_types:
            return entities, types, relations, query_type
        else:
            return entities, relations, query_type
