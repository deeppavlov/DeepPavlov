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


import bisect
import pickle
import unicodedata
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Tuple, List, Union, Dict

import numpy as np
from nltk import word_tokenize
from tqdm import tqdm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

logger = getLogger(__name__)


@register('squad_bert_mapping')
class SquadBertMappingPreprocessor(Component):
    """Create mapping from BERT subtokens to their characters positions and vice versa.
        Args:
            do_lower_case: set True if lowercasing is needed
    """

    def __init__(self, do_lower_case: bool = True, *args, **kwargs):
        self.do_lower_case = do_lower_case

    def __call__(self, contexts, bert_features, *args, **kwargs):
        subtok2chars: List[Dict[int, int]] = []
        char2subtoks: List[Dict[int, int]] = []

        for batch_counter, (context, features) in enumerate(zip(contexts, bert_features)):
            subtokens: List[str]
            if self.do_lower_case:
                context = context.lower()
            if len(args) > 0:
                subtokens = args[0][batch_counter]
            else:
                subtokens = features.tokens
            context_start = subtokens.index('[SEP]') + 1
            idx = 0
            subtok2char: Dict[int, int] = {}
            char2subtok: Dict[int, int] = {}
            for i, subtok in list(enumerate(subtokens))[context_start:-1]:
                subtok = subtok[2:] if subtok.startswith('##') else subtok
                subtok_pos = context[idx:].find(subtok)
                if subtok_pos == -1:
                    # it could be UNK
                    idx += 1  # len was at least one
                else:
                    # print(k, '\t', t, p + idx)
                    idx += subtok_pos
                    subtok2char[i] = idx
                    for j in range(len(subtok)):
                        char2subtok[idx + j] = i
                    idx += len(subtok)
            subtok2chars.append(subtok2char)
            char2subtoks.append(char2subtok)
        return subtok2chars, char2subtoks


@register('squad_bert_ans_preprocessor')
class SquadBertAnsPreprocessor(Component):
    """Create answer start and end positions in subtokens.
        Args:
            do_lower_case: set True if lowercasing is needed
    """

    def __init__(self, do_lower_case: bool = True, *args, **kwargs):
        self.do_lower_case = do_lower_case

    def __call__(self, answers_raw, answers_start, char2subtoks, **kwargs):
        answers, starts, ends = [], [], []
        for answers_raw, answers_start, c2sub in zip(answers_raw, answers_start, char2subtoks):
            answers.append([])
            starts.append([])
            ends.append([])
            for ans, ans_st in zip(answers_raw, answers_start):
                if self.do_lower_case:
                    ans = ans.lower()
                try:
                    indices = {c2sub[i] for i in range(ans_st, ans_st + len(ans)) if i in c2sub}
                    st = min(indices)
                    end = max(indices)
                except ValueError:
                    # 0 - CLS token
                    st, end = 0, 0
                    ans = ''
                starts[-1] += [st]
                ends[-1] += [end]
                answers[-1] += [ans]
        return answers, starts, ends


@register('squad_bert_ans_postprocessor')
class SquadBertAnsPostprocessor(Component):
    """Extract answer and create answer start and end positions in characters from subtoken positions."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, answers_start, answers_end, contexts, bert_features, subtok2chars, *args, **kwargs):
        answers = []
        starts = []
        ends = []
        for batch_counter, (answer_st, answer_end, context, features, sub2c) in \
                enumerate(zip(answers_start, answers_end, contexts, bert_features, subtok2chars)):
            # CLS token is no_answer token
            if answer_st == 0 or answer_end == 0:
                answers += ['']
                starts += [-1]
                ends += [-1]
            else:
                st = self.get_char_position(sub2c, answer_st)
                end = self.get_char_position(sub2c, answer_end)
                if len(args) > 0:
                    subtok = args[0][batch_counter][answer_end]
                else:
                    subtok = features.tokens[answer_end]
                subtok = subtok[2:] if subtok.startswith('##') else subtok
                answer = context[st:end + len(subtok)]
                answers += [answer]
                starts += [st]
                ends += [ends]
        return answers, starts, ends

    @staticmethod
    def get_char_position(sub2c, sub_pos):
        keys = list(sub2c.keys())
        found_idx = bisect.bisect(keys, sub_pos)
        if found_idx == 0:
            return sub2c[keys[0]]

        return sub2c[keys[found_idx - 1]]
