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


@register('squad_iterator')
class SquadIterator(DataLearningIterator):
    """SquadIterator allows to iterate over examples in SQuAD-like datasets.
    SquadIterator is used to train 
    :class:`~deeppavlov.models.torch_bert.torch_transformers_squad:TorchTransformersSquad`.

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
        cqas = []
        if data:
            for article in data['data']:
                for par in article['paragraphs']:
                    context = par['context']
                    for qa in par['qas']:
                        q = qa['question']
                        ans_text = []
                        ans_start = []
                        if qa['answers']:
                            for answer in qa['answers']:
                                ans_text.append(answer['text'])
                                ans_start.append(answer['answer_start'])
                        else:
                            ans_text = ['']
                            ans_start = [-1]
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

    def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None) \
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
            for j in range(i * batch_size, min((i + 1) * batch_size, data_len)):
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
                answer_start = [ans['answer_start']
                                for ans in context['answer']] if len(context['answer']) > 0 else [-1]
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


@register('multi_squad_retr_iterator')
class MultiSquadRetrIterator(DataLearningIterator):
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
        with_answer_rate: sampling rate of contexts with answer
        squad_rate: sampling rate of context from squad dataset (actual rate would be with_answer_rate * squad_rate)

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of ``Random`` initialized with a seed
    """

    def __init__(self, data, seed: Optional[int] = None, shuffle: bool = False,
                 with_answer_rate: Optional[float] = None,
                 squad_rate: Optional[float] = None, *args, **kwargs) -> None:
        self.with_answer_rate = with_answer_rate
        self.squad_rate = squad_rate
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.shuffle = shuffle

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])

        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
        }

        if self.shuffle:
            raise RuntimeError('MultiSquadIterator doesn\'t support shuffling.')

    def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: bool = None) \
            -> Generator[Tuple[Tuple[Tuple[str, str]], Tuple[List[str], List[int]]], None, None]:

        if shuffle is None:
            shuffle = self.shuffle

        if data_type == 'train':
            random = self.np_random
        else:
            random = np.random.RandomState(self.seed)

        if shuffle:
            raise RuntimeError('MultiSquadIterator doesn\'t support shuffling.')

        datafile = self.data[data_type]
        with datafile.open('r', encoding='utf8') as fin:
            end_of_file = False
            while not end_of_file:
                batch = []
                for i in range(batch_size):
                    line = fin.readline()
                    if len(line) == 0:
                        end_of_file = True
                        break

                    qcas = json.loads(line)
                    q = qcas['question']
                    contexts = qcas['contexts']
                    ans_contexts = [c for c in contexts if len(c['answer']) > 0]
                    noans_contexts = [c for c in contexts if len(c['answer']) == 0]
                    ans_clen = len(ans_contexts)
                    noans_clen = len(noans_contexts)
                    # sample context with answer or without answer
                    with_answer_rate = self.with_answer_rate
                    if with_answer_rate is None:
                        with_answer_rate = 1.0 if noans_clen == 0 else ans_clen / (ans_clen + noans_clen)

                    if random.rand() < with_answer_rate or noans_clen == 0:
                        # select random context with answer
                        if self.squad_rate is not None:
                            if random.rand() < self.squad_rate or len(ans_contexts) == 1:
                                # first context is always from squad dataset
                                context = ans_contexts[0]
                            else:
                                context = random.choice(ans_contexts[1:])
                        else:
                            context = random.choice(ans_contexts)
                    else:
                        # select random context without answer
                        # prob ~ context tfidf score
                        # noans_scores = np.array([x['score'] for x in noans_contexts])
                        # noans_scores = noans_scores / np.sum(noans_scores)
                        # context = noans_contexts[np.argmax(random.multinomial(1, noans_scores))]
                        context = random.choice(noans_contexts)

                    answer_text = [ans['text'] for ans in context['answer']] if len(context['answer']) > 0 else ['']
                    answer_start = [ans['answer_start']
                                    for ans in context['answer']] if len(context['answer']) > 0 else [-1]
                    batch.append(((context['context'], q), (answer_text, answer_start)))
                if batch:
                    yield tuple(zip(*batch))

    def get_instances(self, data_type: str = 'train') -> Tuple[Tuple[Tuple[str, str]], Tuple[List[str], List[int]]]:
        data_examples = []
        for f in self.data[data_type]:  # question, contexts, answers
            for line in f.open('r', encoding='utf8'):
                qcas = json.loads(line)
                question = qcas['question']
                for context in qcas['contexts']:
                    answer_text = [x['text'] for x in context['answer']]
                    answer_start = [x['answer_start'] for x in context['answer']]
                    data_examples.append(((context['context'], question), (answer_text, answer_start)))
        return tuple(zip(*data_examples))
