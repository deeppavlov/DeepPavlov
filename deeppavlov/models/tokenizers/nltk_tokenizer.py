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

import nltk

from deeppavlov.core.common.prints import RedirectedPrints
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


with RedirectedPrints():
    nltk.download('punkt')
    nltk.download('stopwords')


@register("nltk_tokenizer")
class NLTKTokenizer(Component):

    def __init__(self, tokenizer="wordpunct_tokenize", download=False, *args, **kwargs):
        if download:
            nltk.download()
        self.tokenizer = getattr(nltk.tokenize, tokenizer, None)
        if not callable(self.tokenizer):
            raise AttributeError("Tokenizer {} is not defined in nltk.tokenizer".format(tokenizer))

    def __call__(self, batch, *args, **kwargs):
        return [self.tokenizer(sent) for sent in batch]
