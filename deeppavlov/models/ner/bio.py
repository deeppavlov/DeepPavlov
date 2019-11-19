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
from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('ner_bio_converter')
class BIOMarkupRestorer(Component):
    """Restores BIO markup for tags batch"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def _convert_to_bio(tags: List[str]) -> List[str]:
        tags_bio = []
        for n, tag in enumerate(tags):
            if tag != 'O':
                if n > 0 and tags[n - 1] == tag:
                    tag = 'I-' + tag
                else:
                    tag = 'B-' + tag
            tags_bio.append(tag)

        return tags_bio

    def __call__(self, tag_batch: List[List[str]], *args, **kwargs) -> List[List[str]]:
        y = [self._convert_to_bio(sent) for sent in tag_batch]
        return y
