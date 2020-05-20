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

from logging import getLogger
import re
from typing import List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import load_pickle

log = getLogger(__name__)


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
        question = self.sanitize(question)
        question_length = len(question)
        entities = []
        types = []
        relations = []
        query_type = ""
        min_length = 100
        for template in self.templates:
            entities_cand = []
            types_cand = []
            template_init = template
            known_entities = re.findall("xxx=\[([a-zа-я\d\s\.-’,]*?)\]", template)
            if known_entities:
                entities_cand += known_entities
                for known_ent in known_entities:
                    template = template.replace(f"xxx=[{known_ent}]", known_ent)
            
            known_types = re.findall("ttt=\[([a-zа-я\d\s\.-’,]*?)\]", template)
            if known_types:
                types_cand += known_types
                for known_type in known_types:
                    template = template.replace(f"ttt=[{known_type}]", known_type)
            
            if all([not template.startswith(tok) for tok in ["xxx", "ttt", "yyy"]]):
                template = "yyy" + template
            if all([not template.endswith(tok) for tok in ["xxx?", "ttt?", "yyy?"]]):
                template = template.replace('?', 'yyy?')
            template_len = len(template.replace('xxx', '').replace('ttt', '').replace('yyy', '')) - \
                  sum([len(entity) for entity in entities_cand]) - sum([len(entity_type) for entity_type in types_cand])
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
                entities_cand += [fnd[0][pos].replace('?', '') for pos in positions_entity_tokens]
                types_cand += [fnd[0][pos].replace('?', '').split(',')[0] for pos in positions_type_tokens]
                unuseful_tokens = [fnd[0][pos].replace('?', '') for pos in positions_unuseful_tokens]
                entity_lengths = [len(entity) for entity in entities_cand]
                entity_num_tokens = all([len(entity.split(' ')) < 6 for entity in entities_cand])
                type_lengths = [len(entity_type) for entity_type in types_cand]
                unuseful_tokens_len = sum([len(unuseful_tok) for unuseful_tok in unuseful_tokens])
                log.debug(f"found template: {template_init}, {template_regexp}, {fnd}")

                if 0 not in entity_lengths or 0 not in type_lengths and entity_num_tokens:
                    cur_len = sum(entity_lengths) + sum(type_lengths)
                    if cur_len < min_length and unuseful_tokens_len + template_len + cur_len == question_length:
                        entities = [entity.replace("the uk", "united kingdom").replace("the us", "united states") for entity in entities_cand]
                        types = types_cand
                        relations = self.templates[template_init][1:]
                        query_type = self.templates[template_init][0]
                        min_length = cur_len

        if self.return_types:
            return entities, types, relations, query_type
        else:
            return entities, relations, query_type

    def sanitize(self, question: str) -> str:
        if question.startswith("the "):
            question = question[4:]
        if question.startswith("a "):
            question = question[2:]
        
        date_interval = re.findall("([\d]{4}-[\d]{4})", question)
        if date_interval:
            question = question.replace(date_interval[0], '')
        question = question.replace('  ', ' ')
        return question
