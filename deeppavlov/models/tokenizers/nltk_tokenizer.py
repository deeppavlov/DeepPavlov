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


import nltk

from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.registry import register

@register("nltk_tokenizer")
class NLTKTokenizer(Inferable):

    def __init__(self,  tokenizer="wordpunct_tokenize", *args, **kwargs):
        super().__init__()
        nltk.download()
        self.tokenizer = getattr(nltk.tokenize, tokenizer, None)
        if not callable(self.tokenizer):
            raise AttributeError("Tokenizer {} is not defined in nltk.tokenizer".format(tokenizer))

    def infer(self, instance, *args, **kwargs):
        if type(instance) is str:
            return " ".join(self.tokenizer(instance))

        elif type(instance) is list:
            tokenized_batch = []
            for text in instance:
                tokenized_batch.append(" ".join(self.tokenizer(text)))
            return tokenized_batch
