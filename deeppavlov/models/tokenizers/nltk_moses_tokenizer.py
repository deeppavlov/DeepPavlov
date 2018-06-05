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

from nltk.tokenize.moses import MosesDetokenizer, MosesTokenizer

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register("nltk_moses_tokenizer")
class NLTKTokenizer(Component):

    def __init__(self, escape=False, *args, **kwargs):
        self.escape = escape
        self.tokenizer = MosesTokenizer()
        self.detokenizer = MosesDetokenizer()

    def __call__(self, batch, *args, **kwargs):
        if isinstance(batch[0], str):
            return [self.tokenizer.tokenize(line, escape=self.escape) for line in batch]
        else:
            return [self.detokenizer.detokenize(line, return_str=True, unescape=self.escape)
                    for line in batch]
