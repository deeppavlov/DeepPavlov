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
from logging import getLogger
from typing import List, Dict

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

logger = getLogger(__name__)


@register('squad_bert_mapping')
class SquadBertMappingPreprocessor(Component):
    """Create mapping from BERT subtokens to their characters positions and vice versa.
        Args:
            do_lower_case: set True if lowercasing is needed
    """

    def __init__(self, do_lower_case: bool = True, *args, **kwargs):
        self.do_lower_case = do_lower_case

    def __call__(self, contexts_batch, bert_features_batch, subtokens_batch, **kwargs):
        subtok2chars_batch: List[List[Dict[int, int]]] = []
        char2subtoks_batch: List[List[Dict[int, int]]] = []

        for batch_counter, (context_list, features_list, subtokens_list) in \
                enumerate(zip(contexts_batch, bert_features_batch, subtokens_batch)):
            subtok2chars_list, char2subtoks_list = [], []
            for context, features, subtokens in zip(context_list, features_list, subtokens_list):
                if self.do_lower_case:
                    context = context.lower()
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
                subtok2chars_list.append(subtok2char)
                char2subtoks_list.append(char2subtok)
            subtok2chars_batch.append(subtok2chars_list)
            char2subtoks_batch.append(char2subtoks_list)
        return subtok2chars_batch, char2subtoks_batch


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
                    indices = {c2sub[0][i] for i in range(ans_st, ans_st + len(ans)) if i in c2sub[0]}
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

    def __call__(self, answers_start_batch, answers_end_batch, contexts_batch,
                 subtok2chars_batch, subtokens_batch, ind_batch, *args, **kwargs):
        answers = []
        starts = []
        ends = []
        for answer_st, answer_end, context_list, sub2c_list, subtokens_list, ind in \
                zip(answers_start_batch, answers_end_batch, contexts_batch, subtok2chars_batch, subtokens_batch,
                    ind_batch):
            sub2c = sub2c_list[ind]
            subtok = subtokens_list[ind][answer_end]
            context = context_list[ind]
            # CLS token is no_answer token
            if answer_st == 0 or answer_end == 0:
                answers += ['']
                starts += [-1]
                ends += [-1]
            else:
                st = self.get_char_position(sub2c, answer_st)
                end = self.get_char_position(sub2c, answer_end)

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
