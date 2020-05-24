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
    """SquadIteratorTwitter allows to iterate over examples in TweetQA dataset.
    SquadIterator is used to train :class:`~deeppavlov.models.squad.squad.SquadModel`.

    It extracts ``context``, ``question``, ``answer_text`` and ``answer_start`` position from dataset.
    Example from a dataset is a tuple of ``(context, question)`` and ``(answer_text, answer_start)``

    Attributes:
        train: train examples
        valid: validation examples
        test: test examples

    """

    def preprocess(self, data: List[Dict[str, Any]], *args, **kwargs) -> \
            List[Tuple[Tuple[str, str], Tuple[List[str], List[int]]]]:
        """Extracts context, question, answer, answer_start from TweetQA data

        Args:
            data: data in squad format

        Returns:
            list of (context, question), (answer_text, answer_start)
            answer text and answer_start are lists
        """
        cqas = []
        for item in data:
            
            context = item['Tweet'].lower()
            q = item['Question'].lower()

            # No answer case
            if 'Answer' not in item:
                cqas.append(((context, q), ([''], [-1])))
                break

            found = False
            for ans_text in map(lambda x: x.lower(), item['Answer']): 
                if ans_text in context:
                    ans_start = context.find(ans_text)
                    cqas.append(((context, q), ([ans_text], [ans_start])))
                    found = True
                    break
            # there were some answers, but none of them came from context
            if not found:
                cqas.append(((context, q), ([''], [-1])))

        return cqas