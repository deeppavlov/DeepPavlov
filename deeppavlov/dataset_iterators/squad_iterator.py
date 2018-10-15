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


from typing import Dict, Any, List, Tuple, Generator, Optional

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('squad_iterator')
class SquadIterator(DataLearningIterator):
    """SquadIterator allows to iterate over examples in SQuAD-like datasets.
    SquadIterator is used to train :class:`~deeppavlov.models.squad.squad.SquadModel`.

    It extracts ``context``, ``question``, ``answer_text`` and ``answer_start`` position from dataset.
    Example from a dataset is a tuple of ``(context, question)`` and ``(answer_text, answer_start)``

    Attributes:
        train: train examples
        valid: validation examples
        test: test examples

    """

    def split(self, *args, **kwargs) -> None:
        for dt in ['train', 'valid', 'test']:
            setattr(self, dt, SquadIterator._extract_cqas(getattr(self, dt)))

    @staticmethod
    def _extract_cqas(data: Dict[str, Any]) -> List[Tuple[Tuple[str, str], Tuple[List[str], List[int]]]]:
        """Extracts context, question, answer, answer_start from SQuAD data

        Args:
            data: data in squad format

        Returns:
            list of (context, question), (answer_text, answer_start)
            answer text and answer_start are lists

        """
        cqas = []
        if data:
            for article in data['data']:
                for par in article['paragraphs']:
                    context = par['context']
                    for qa in par['qas']:
                        q = qa['question']
                        ans_text = []
                        ans_start = []
                        for answer in qa['answers']:
                            ans_text.append(answer['text'])
                            ans_start.append(answer['answer_start'])
                        cqas.append(((context, q), (ans_text, ans_start)))
        return cqas


@register('multi_squad_iterator')
class MultiSquadIterator(DataLearningIterator):
    """Dataset iterator for multiparagraph-SQuAD dataset.

    With ``with_answer_rate`` rate samples context with answer and with ``1 - with_answer_rate`` samples context
    from the same article, but without an answer. Contexts without an answer are sampled according to
    their tfidf scores (tfidf score between question and context).

    It extracts ``context``, ``question``, ``answer_text`` and ``answer_start`` position from dataset.
    Example from a dataset is a tuple of ``(context, question)`` and ``(answer_text, answer_start)``. If there is
    no answer in context, then ``answer_text`` is empty string and `answer_start` is equal to -1.

    Args:
        data: dict with keys ``'train'``, ``'valid'`` and ``'test'`` and values
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching
        with_answer_rate: sampling rate of contexts with answer

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of ``Random`` initialized with a seed
    """

    def __init__(self, data, seed: Optional[int] = None, shuffle: bool = True, with_answer_rate: float = 0.666,
                 *args, **kwargs) -> None:
        self.with_answer_rate = with_answer_rate
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        super().__init__(data, seed, shuffle, *args, **kwargs)

    def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None)\
            -> Generator[Tuple[Tuple[Tuple[str, str]], Tuple[List[str], List[int]]], None, None]:

        if shuffle is None:
            shuffle = self.shuffle

        if data_type == 'train':
            random = self.np_random
        else:
            random = np.random.RandomState(self.seed)

        if shuffle:
            random.shuffle(self.data[data_type])

        data = self.data[data_type]
        data_len = len(data)

        for i in range((data_len - 1) // batch_size + 1):
            batch = []
            for j in range(i * batch_size, min((i+1) * batch_size, data_len)):
                q = data[j]['question']
                contexts = data[j]['contexts']
                ans_contexts = [c for c in contexts if len(c['answer']) > 0]
                noans_contexts = [c for c in contexts if len(c['answer']) == 0]
                # sample context with answer or without answer
                if random.rand() < self.with_answer_rate or len(noans_contexts) == 0:
                    # select random context with answer
                    context = random.choice(ans_contexts)
                else:
                    # select random context without answer
                    # prob ~ context tfidf score
                    noans_scores = np.array([x['score'] for x in noans_contexts])
                    noans_scores = noans_scores / np.sum(noans_scores)
                    context = noans_contexts[np.argmax(random.multinomial(1, noans_scores))]

                answer_text = [ans['text'] for ans in context['answer']] if len(context['answer']) > 0 else ['']
                answer_start = [ans['answer_start'] for ans in context['answer']] if len(context['answer']) > 0 else [-1]
                batch.append(((context['context'], q), (answer_text, answer_start)))
            yield tuple(zip(*batch))

    def get_instances(self, data_type: str = 'train') -> Tuple[Tuple[Tuple[str, str]], Tuple[List[str], List[int]]]:
        data_examples = []
        for qcas in self.data[data_type]:  # question, contexts, answers
            question = qcas['question']
            for context in qcas['contexts']:
                answer_text = [x['text'] for x in context['answer']]
                answer_start = [x['answer_start'] for x in context['answer']]
                data_examples.append(((context['context'], question), (answer_text, answer_start)))
        return tuple(zip(*data_examples))