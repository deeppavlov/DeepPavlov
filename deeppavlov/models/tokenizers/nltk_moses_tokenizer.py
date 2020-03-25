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
from typing import Union, List

from sacremoses import MosesDetokenizer, MosesTokenizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register("nltk_moses_tokenizer")
class NLTKMosesTokenizer(Component):
    """Class for splitting texts on tokens using NLTK wrapper over MosesTokenizer

    Attributes:
        escape: whether escape characters for use in html markup
        tokenizer: tokenizer instance from nltk.tokenize.moses
        detokenizer: detokenizer instance from nltk.tokenize.moses

    Args:
        escape: whether escape characters for use in html markup
    """

    def __init__(self, escape: bool = False, *args, **kwargs):
        self.escape = escape
        self.tokenizer = MosesTokenizer()
        self.detokenizer = MosesDetokenizer()

    def __call__(self, batch: List[Union[str, List[str]]]) -> List[Union[List[str], str]]:
        """Tokenize given batch of strings or detokenize given batch of lists of tokens

        Args:
            batch: list of text samples or list of lists of tokens

        Returns:
            list of lists of tokens or list of text samples
        """
        if isinstance(batch[0], str):
            return [self.tokenizer.tokenize(line, escape=self.escape) for line in batch]
        else:
            return [self.detokenizer.detokenize(line, return_str=True, unescape=self.escape)
                    for line in batch]
