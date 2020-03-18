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
import pickle
from typing import List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable


@register('template_matcher')
class TemplateMatcher(Component, Serializable):
    """
        This class matches the question with one of the templates
        to extract entity substrings and define which relations
        corresponds to the question
    """

    def __init__(self, load_path: str,
                 templates_filename: str = None,
                 **kwargs) -> None:
        """

        Args:
            load_path: path to folder with file with templates
            templates_filename: file with templates
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self._templates_filename = templates_filename
        self.load()

    def load(self) -> None:
        if self._templates_filename is not None:
            with open(self.load_path / self._templates_filename, 'rb') as t:
                self.templates = pickle.load(t)

    def save(self) -> None:
        raise NotImplementedError

    def __call__(self, question: str, *args, **kwargs) -> Tuple[List[str], List[Tuple[str]], str]:
        question = question.lower()
        question_length = len(question)
        entities = []
        relations = []
        query_type = ""
        min_length = 100
        for template in self.templates:
            template_len = len(template.replace('xxx', ''))
            template_regexp = template.replace("xxx", "([a-zа-я\d\s\.]+)")
            fnd = re.findall(template_regexp, question)
            if fnd:
                entities_cand = fnd[0]
                if str(type(entities_cand)) == "<class 'str'>":
                    entities_cand = [entities_cand]

                found = True
                entity_lengths = [len(entity) for entity in entities_cand]
                for length in entity_lengths:
                    if length == 0:
                        found = False
                if found:
                    cur_len = sum(entity_lengths)
                    if cur_len < min_length and template_len + cur_len == question_length:
                        entities = entities_cand
                        relations = self.templates[template][1:]
                        query_type = self.templates[template][0]
                        min_length = cur_len

        return entities, relations, query_type
