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

from typing import List, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

StrTokenReverserInfo = Union[List[str], List['StrTokenReverserInfo']]


@register('str_token_reverser')
class StrTokenReverser(Component):
    """Component for converting strings to strings with reversed token positions

    Args:
        tokenized: The parameter is only needed to reverse tokenized strings.
    """

    def __init__(self, tokenized: bool = False, *args, **kwargs) -> None:
        self.tokenized = tokenized

    @staticmethod
    def _reverse_str(raw_string):
        splitted = raw_string.split()
        splitted.reverse()
        string = ' '.join(splitted)
        return string

    @staticmethod
    def _reverse_tokens(raw_tokens):
        raw_tokens.reverse()
        return raw_tokens

    def __call__(self, batch: Union[str, list, tuple]) -> StrTokenReverserInfo:
        """Recursively search for strings in a list and convert them to strings with reversed token positions

        Args:
            batch: a string or a list containing strings

        Returns:
            the same structure where all strings tokens are reversed
        """
        if isinstance(batch, (list, tuple)):
            batch = batch.copy()

        if self.tokenized:
            if isinstance(batch, (list, tuple)):
                if isinstance(batch[-1], str):
                    return self._reverse_tokens(batch)
                else:
                    return [self(line) for line in batch]
            raise RuntimeError(f'The objects passed to the reverser are not list or tuple! '
                               f' But they are {type(batch)}.'
                               f' If you want to passed str type directly use option tokenized = False')
        else:
            if isinstance(batch, (list, tuple)):
                return [self(line) for line in batch]
            else:
                return self._reverse_str(batch)
