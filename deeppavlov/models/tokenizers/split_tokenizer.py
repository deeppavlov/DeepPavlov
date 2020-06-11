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

from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register("split_tokenizer")
class SplitTokenizer(Component):
    """
    Generates utterance's tokens by mere python's ``str.split()``.

    Doesn't have any parameters.
    """

    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, batch: List[str]) -> List[List[str]]:
        """
        Tokenize given batch

        Args:
            batch: list of texts to tokenize

        Returns:
            tokenized batch
        """
        if isinstance(batch, (list, tuple)):
            return [sample.split() for sample in batch]
        else:
            raise NotImplementedError('not implemented for types other than'
                                      ' list or tuple')
