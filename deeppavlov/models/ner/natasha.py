"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



from typing import List, Generator, Any
from collections import defaultdict

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger

import natasha

logger = get_logger(__name__)


@register('natasha_ner')
class NatashaNer(Component):
    def __init__(self, classes,  output_format='slots', *args, **kwargs):
        self.extractors = []
        self.output_format = output_format
        for c in classes:
            self.extractors.append(getattr(natasha, c)())

    def __call__(self, batch, *args, **kwargs):
        if isinstance(batch, (list, tuple)):
            return [self._process(b) for b in batch]
        else:
            return self._process(batch)

    def _process(self, utt):
        result = defaultdict(list)
        for extractor in self.extractors:
            ms = extractor(utt)
            for m in ms:
                v = m
                if self.output_format == 'facts':
                    v = v.fact
                if self.output_format == 'slots':
                    v = dict(v.fact.as_json)
                result[extractor.__class__.__name__.lower().replace("extractor", "")].append(v)
        return result
