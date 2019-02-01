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

from typing import Tuple, Optional, List

import re

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.skill.skill import Skill

proposals = {
    'en': 'expecting_arg: {}',
    'ru': 'Пожалуйста, введите параметр {}'
}


class DefaultStatelessSkillWithHistory(Skill):
    """Default stateless skill class.

    The class is intended to be used for as a default skill wrapping DeepPavlov models.

    Attributes:
        model: DeepPavlov model to be wrapped into default skill instance.
    """
    def __init__(self, model: Chainer, lang: str='en', *args, **kwargs) -> None:
        self.model = model
        self.proposal: str = proposals[lang]

        # One can choose the context depth dynamically
        # by typing the command: ".3", where 3 is a new desired context length
        self.num_turns = 2

    def __call__(self, utterances_batch: list, history_batch: list,
                 states_batch: Optional[list]=None) -> Tuple[list, list, list]:
        """Returns skill inference result.

        Returns batches of skill inference results, estimated confidence
            levels and up to date states corresponding to incoming utterance
            batch. Also handles interaction with multiargument models using
            skill states.

        Args:
            utterances_batch: A batch of utterances of any type.
            history_batch: Not used. A batch of list typed histories for each
                utterance.
            states_batch: A batch of states for each utterance.

        Returns:
            response: A batch of arbitrary typed skill inference results.
            confidence: A batch of float typed confidence levels for each of
                skill inference result.
            states: Optional. A batch of states for each response.
        """
        batch_len = len(utterances_batch)
        confidence_batch = [1.0] * batch_len

        response_batch: List[Optional[str]] = [None] * batch_len
        infer_indexes = []

        if not states_batch:
            states_batch: List[Optional[dict]] = [None] * batch_len

        for utt_i, utterance in enumerate(utterances_batch):
            if not states_batch[utt_i]:
                states_batch[utt_i] = {'expected_args': list(self.model.in_x), 'received_values': []}

            if utterance:
                states_batch[utt_i]['expected_args'].pop(0)

                # dynamically adjust context depth
                if len(utterance) == 2:
                    if utterance.startswith('.') and utterance[1].isdigit():
                        self.num_turns = int(utterance[1])
                        # if self.num_turns < 1: self.num_turns = 1
                        utterance = ''

                if self.num_turns != 0:
                    #### Append history to the utterance ####
                    expanded_context = self._expand_context(history_batch[utt_i], padding="pre")
                    exp_context_str = " & ".join(expanded_context)
                    exp_context_str += " & {}".format(utterance)
                    #########################################
                else:
                    exp_context_str = "&&&&&&&&&" + utterance.replace('&', '.')
                # print('[utterance]:', utterance, "[input]: ", exp_context_str, "[num_turns]:", self.num_turns)

                states_batch[utt_i]['received_values'].append(exp_context_str)

            if states_batch[utt_i]['expected_args']:
                response = self.proposal.format(states_batch[utt_i]['expected_args'][0])
                response_batch[utt_i] = response
            else:
                infer_indexes.append(utt_i)

        if infer_indexes:
            infer_utterances = zip(*[tuple(states_batch[i]['received_values']) for i in infer_indexes])
            infer_results = self.model(*infer_utterances)

            if len(self.model.out_params) > 1:
                infer_results = ['; '.join([str(out_y) for out_y in result]) for result in zip(*infer_results)]

            for infer_i, infer_result in zip(infer_indexes, infer_results):
                response_batch[infer_i] = infer_result
                states_batch[infer_i] = None

        return response_batch, confidence_batch, states_batch


    def _expand_context(self, context: List[str], padding: str) -> List[str]:
        """
        Align context length by using pre/post padding of empty sentences up to ``self.num_turns`` sentences
        or by reducing the number of context sentences to ``self.num_turns`` sentences.

        Args:
            context (List[str]): list of raw context sentences
            padding (str): "post" or "pre" context sentences padding

        Returns:
            List[str]: list of ``self.num_turns`` context sentences
        """
        if padding == "post":
            sent_list = context
            res = sent_list + (self.num_turns - len(sent_list)) * \
                  [''] if len(sent_list) < self.num_turns else sent_list[:self.num_turns]
            return res
        elif padding == "pre":
            # context[-(self.num_turns + 1):-1]  because the last item of `context` is always '' (empty string)
            sent_list = context[-self.num_turns:]
            if len(sent_list) <= self.num_turns:
                tmp = sent_list[:]
                sent_list = [''] * (self.num_turns - len(sent_list))
                sent_list.extend(tmp)
            return sent_list