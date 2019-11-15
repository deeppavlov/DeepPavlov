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

from nltk import word_tokenize
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('lazy_tokenizer')
class LazyTokenizer(Component):
    """Tokenizes if there is something to tokenize."""

    def __init__(self, **kwargs):
        pass

    @overrides
    def __call__(self, batch, *args, **kwargs):
        if len(batch) > 0 and isinstance(batch[0], str):
            batch = [word_tokenize(utt) for utt in batch]
        return batch
