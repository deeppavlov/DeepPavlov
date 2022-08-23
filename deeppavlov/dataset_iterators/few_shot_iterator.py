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


import json
from typing import Dict, Any, List, Tuple, Generator, Optional

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('few_shot_iterator')
class FewShotIterator(DataLearningIterator):
    """Dataset iterator for multiparagraph-SQuAD dataset.

    reads data from jsonl files

    With ``with_answer_rate`` rate samples context with answer and with ``1 - with_answer_rate`` samples context
    from the same article, but without an answer. Contexts without an answer are sampled from uniform distribution.
    If ``with_answer_rate`` is None than we compute actual ratio for each data example.

    It extracts ``context``, ``question``, ``answer_text`` and ``answer_start`` position from dataset.
    Example from a dataset is a tuple of ``(context, question)`` and ``(answer_text, answer_start)``. If there is
    no answer in context, then ``answer_text`` is empty string and `answer_start` is equal to -1.

    Args:
        data: dict with keys ``'train'``, ``'valid'`` and ``'test'`` and values
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of ``Random`` initialized with a seed
    """

    def __init__(self, data, *args, **kwargs) -> None:

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])

        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
        }


    def gen_batches(self, batch_size: int, data_type: str, shuffle: bool):
        train_examples = self.data['train']
        test_examples = self.data[data_type]

        for test_example, test_cat in test_examples:
            batch = []
            for train_example, train_cat in train_examples:

                if isinstance(train_cat, list) or isinstance(train_cat, tuple):
                    train_cat = train_cat[0]
                
                if isinstance(test_cat, list) or isinstance(test_cat, tuple):
                    test_cat = test_cat[0]
                

                batch.append(((train_example, test_example, train_cat), test_cat))

            if batch:
                yield tuple(zip(*batch))
