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

import sys
import json
from typing import Dict, Any, List, Tuple, Generator, Optional

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('squad_iterator_tweetqa')
class SquadIteratorTwitter(DataLearningIterator):
    """SquadIterator allows to iterate over examples in SQuAD-like datasets.
    SquadIterator is used to train :class:`~deeppavlov.models.squad.squad.SquadModel`.

    It extracts ``context``, ``question``, ``answer_text`` and ``answer_start`` position from dataset.
    Example from a dataset is a tuple of ``(context, question)`` and ``(answer_text, answer_start)``

    Attributes:
        train: train examples
        valid: validation examples
        test: test examples

    """

    def preprocess(self, data: Dict[str, Any], *args, **kwargs) -> \
            List[Tuple[Tuple[str, str], Tuple[List[str], List[int]]]]:
        """Extracts context, question, answer, answer_start from SQuAD data

        Args:
            data: data in squad format

        Returns:
            list of (context, question), (answer_text, answer_start)
            answer text and answer_start are lists

        """
        
        counter_nonzero = 0
        counter_notfound = 0
        
        cqas = []
        if data:
            for item in data:
                
                context = item['Tweet'].lower()
                q = item['Question'].lower()
                
                if 'Answer' in item:
#                     if len(item['Answer']) != 1:
#                         counter_nonzero += 1
                    
                    for ans_text in item['Answer']:
                        
                        ans_text = ans_text.lower()
                    
                        if ans_text in context:
                            ans_start = context.find(ans_text)
                            cqas.append(((context, q), ([ans_text], [ans_start])))
                            break
#                         else:
#                             counter_notfound += 1
#                             print(item)
#                             print("---------------")

                else:
                    cqas.append(((context, q), ([''], [-1])))
#                     counter_nonzero += 1
        
#         print('counter_nonzero', counter_nonzero)
#         print(counter_notfound)
        print(len(cqas))

        return cqas
    