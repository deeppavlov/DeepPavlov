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

import pickle
from string import punctuation
from typing import List, Tuple, Optional, Dict

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable


@register('template_matcher')
class TemplateMatcher(Component, Serializable):
    def __init__(self, load_path: str, templates_filename: str = None, *args, **kwargs) -> None:
        super().__init__(save_path=None, load_path=load_path)
        self._templates_filename = templates_filename
        self.load()

    def load(self) -> None:
        if self._templates_filename is not None:
            with open(self.load_path.parent / self._templates_filename, 'rb') as t:
                self.templates = pickle.load(t)

    def save(self) -> None:
        pass

    def call(self, tokens: List[str], *args, **kwargs) -> Tuple[str, str]:
        s_sanitized = ' '.join([ch for ch in tokens if ch not in punctuation]).lower()
        s_sanitized = s_sanitized.replace("'", '').replace("`", '')
        ent = ''
        relation = ''
        for template in self.templates:
            template_start, template_end = template.lower().split('xxx')
            if template_start in s_sanitized and template_end in s_sanitized:
                template_start_pos = s_sanitized.find(template_start)
                template_end_pos = s_sanitized.find(template_end)
                ent_cand = s_sanitized[template_start_pos+len(template_start): template_end_pos or len(s_sanitized)]
                if len(ent_cand) < len(ent) or len(ent) == 0:
                    ent = ent_cand
                    relation = self.templates[template]
        return ent, relation
