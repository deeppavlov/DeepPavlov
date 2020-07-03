# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
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

import jieba

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register("jieba_tokenizer")
class JiebaTokenizer(Component):
    """
    Tokenizes chinese text into tokens
    
    Doesn't have any parameters.
    """

    def __init__(self, **kwargs) -> None:
        jieba.initialize()
        pass

    @staticmethod
    def tokenize_str(text: str) -> str:
        """
        Tokenize a single string

        Args:
            text: a string to tokenize

        Returns:
            tokenized string
        """
        return ' '.join(jieba.cut(text))

    def __call__(self, batch: Union[List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        """
        Tokenize either list of strings or list of list of strings

        Args:
            batch a list of either strings or list of strings

        Returns:
            tokenized strings in the given format
        """

        if isinstance(batch[0], str):
            batch_tokenized = [JiebaTokenizer.tokenize_str(s) for s in batch]
        elif isinstance(batch[0], list):
            for lst in batch:
                batch_tokenized = [self(lst) for lst in batch]
        else:
            raise NotImplementedError('Not implemented for types other than'
                                      ' str or list')

        return batch_tokenized
