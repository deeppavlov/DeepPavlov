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

from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
import re

from .feb_objects import *
from .feb_common import FebComponent

from question2wikidata import questions, functions


log = get_logger(__name__)


@register('text_parser')
class FebTextParser(FebComponent):
    """Convert batch of strings
    sl = ["author_birthplace author Лев Николаевич Толстой",
      -(to)->
        utterence object
      """
    @classmethod
    def component_type(cls):
        return cls.START_COMPONENT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def test_and_prepare(self, utt):
        question = utt.text

        razdel_tokens = FebToken.tokenize(question)
        # razdel_tokens = [t for t in tokens if t.type != FebToken.PUNCTUATION]
        stemmer_tokens = FebToken.stemmer(question)

        var_dump(header='razdel_tokens', msg = razdel_tokens)
        var_dump(header='stemmer_tokens', msg = stemmer_tokens)

        
        outer_index, inner_index = 0, 0
        inner_shift = 0
        for razdel_token in razdel_tokens:
            for inner_index, stemmer_token in enumerate(stemmer_tokens):
                if inner_index < inner_shift: continue
                if razdel_token == stemmer_token:
                    razdel_token.set_pos(stemmer_token.pos)
                    razdel_token.set_normal_form(stemmer_token.normal_form)
                    inner_shift = inner_index
                    break
            else:
                razdel_token.set_pos('X')
        var_dump(header='razdel_tokens with PoS', msg = razdel_tokens)           

        utt.tokens = razdel_tokens
        
        tokens = [t for t in utt.tokens if t.type != FebToken.PUNCTUATION]

        return [(utt, {})]

