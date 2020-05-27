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
import multiprocessing as mp
import json
import functools
from logging import getLogger
from typing import Tuple, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)


class RegexpMatcher:
    def __init__(self, question):
        self.question = question

    def __call__(self, template):
        res = re.findall(template["template_regexp"], self.question)
        found_template = []
        if res:
            found_template.append((res[0], template))
        return found_template


@register('template_matcher')
class TemplateMatcher(Serializable):
    """
        This class matches the question with one of the templates
        to extract entity substrings and define which relations
        corresponds to the question
    """

    def __init__(self, load_path: str, templates_filename: str,
                 num_processors: int = None, **kwargs) -> None:
        """

        Args:
            load_path: path to folder with file with templates
            templates_filename: file with templates
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.templates_filename = templates_filename
        self.num_processors = mp.cpu_count() if num_processors == None else num_processors
        self.pool = mp.Pool(self.num_processors)
        self.load()

    def load(self) -> None:
        log.debug(f"(load)self.load_path / self.templates_filename: {self.load_path / self.templates_filename}")
        with open(self.load_path / self.templates_filename) as fl:
            self.templates = json.load(fl)

    def save(self) -> None:
        raise NotImplementedError

    def __call__(self, question: str) -> Tuple[List[str], List[str], List[Tuple[str]], List[str], str]:
        question = question.lower()
        question = self.sanitize(question)
        question_length = len(question)
        entities, types, relations, relation_dirs = [], [], [], []
        query_type = ""
        results = self.pool.map(RegexpMatcher(question), self.templates)
        results = functools.reduce(lambda x, y: x + y, results)
        replace_tokens = [("the uk", "united kingdom"), ("the us", "united states")]
        if results:
            min_length = 100
            for result in results:
                found_ent, template = result
                positions_entity_tokens = template["positions_entity_tokens"]
                positions_type_tokens = template["positions_type_tokens"]
                positions_unuseful_tokens = template["positions_unuseful_tokens"]
                template_len = template["template_len"]
                entities_cand = [found_ent[pos].replace('?', '') for pos in positions_entity_tokens]
                types_cand = [found_ent[pos].replace('?', '').split(',')[0] for pos in positions_type_tokens]
                unuseful_tokens = [found_ent[pos].replace('?', '') for pos in positions_unuseful_tokens]
                entity_lengths = [len(entity) for entity in entities_cand]
                entity_num_tokens = all([len(entity.split(' ')) < 6 for entity in entities_cand])
                type_lengths = [len(entity_type) for entity_type in types_cand]
                unuseful_tokens_len = sum([len(unuseful_tok) for unuseful_tok in unuseful_tokens])
                log.debug(f"found template: {template}, {found_ent}")

                if 0 not in entity_lengths or 0 not in type_lengths and entity_num_tokens:
                    cur_len = sum(entity_lengths) + sum(type_lengths)
                    log.debug(f"lengths: entity+type {cur_len}, question {question_length}, "
                              f"template {template_len}, unuseful tokens {unuseful_tokens_len}")
                    if cur_len < min_length and unuseful_tokens_len + template_len + cur_len == question_length:
                        entities = entities_cand
                        for old_token, new_token in replace_tokens:
                            entities = [entity.replace(old_token, new_token) for entity in entities]
                        types = types_cand
                        relations = template["relations"]
                        relation_dirs = template["rel_dirs"]
                        query_type = template["template_type"]
                        min_length = cur_len

        return entities, types, relations, relation_dirs, query_type

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
