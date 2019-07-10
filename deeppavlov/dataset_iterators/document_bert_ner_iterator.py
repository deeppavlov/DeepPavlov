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

from copy import copy
from random import Random
from typing import List, Dict, Tuple, Any, Iterator, Optional
import itertools
from logging import getLogger

import numpy as np
from bert_dp.preprocessing import convert_examples_to_features, InputExample, InputFeatures
from bert_dp.tokenization import FullTokenizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component

logger = getLogger(__name__)


@register('document_bert_ner_iterator')
class DocumentBertNerIterator(DataLearningIterator):
    """Dataset iterator for learning models, e. g. neural networks.

    Args:
        data: list of (x, y) pairs for every data type in ``'train'``, ``'valid'`` and ``'test'``
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of ``Random`` initialized with a seed
    """

    def __init__(self,
                 data: Dict[str, List[Tuple[Any, Any]]],
                 bert_tokenizer_vocab_file: str,
                 do_lower_case: bool = False,
                 left_context_rate: float = 0.5,
                 max_seq_length: int = None,
                 one_sample_per_doc: bool = False,
                 seed: int = None,
                 shuffle: bool = True,
                 *args, **kwargs) -> None:
        self.max_seq_length = max_seq_length or float('inf')
        self.one_sample_per_doc = one_sample_per_doc
        self.left_context_rate = left_context_rate
        vocab_file = str(expand_path(bert_tokenizer_vocab_file))
        self.tokenizer = FullTokenizer(vocab_file=vocab_file,
                                       do_lower_case=do_lower_case)
        super().__init__(data, seed, shuffle, *args, **kwargs)

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None) -> Iterator[Tuple[tuple, tuple]]:
        """Generate batches of inputs and expected output to train neural networks

        Args:
            batch_size: number of samples in batch
            data_type: can be either 'train', 'test', or 'valid'
            shuffle: whether to shuffle dataset before batching

        Yields:
             a tuple of a batch of inputs and a batch of expected outputs
        """
        if shuffle is None:
            shuffle = self.shuffle

        data = self.data[data_type]
        # doc_data: list of tuples (doc_id, list of doc samples)
        doc_data = [(doc_id, [self.rm_doc_id(s) for s in doc])
                    for doc_id, doc in itertools.groupby(data, key=self.get_doc_id)]
        num_docs = len(doc_data)

        if num_docs == 0:
            return

        # get all sentences from document
        doc_chunks = [self.chunks_from_doc(doc) for doc_id, doc in doc_data]
        if self.one_sample_per_doc:
            samples = [next(chunk) for chunk in doc_chunks]
        else:
            samples = [s for chunk in doc_chunks for s in chunk]
        num_samples = len(samples)

        order = list(range(num_samples))

        if shuffle:
            self.random.shuffle(order)

        if batch_size < 0:
            batch_size = num_samples

        for i in range((num_samples - 1) // batch_size + 1):
            yield tuple(zip(*[samples[o]
                              for o in order[i * batch_size: (i + 1) * batch_size]]))

    def get_instances(self, data_type: str = 'train') -> Tuple[tuple, tuple]:
        data = self.data[data_type]
        data_wo_doc_ids = (self.rm_doc_id(s) for s in data)
        return tuple(zip(*data_wo_doc_ids))

    @staticmethod
    def get_doc_id(sample: Tuple[Any, Any]) -> int:
        return sample[0][-1]

    @staticmethod
    def rm_doc_id(sample: Tuple[Any, Any]) -> Tuple[Any, Any]:
        x, y = sample
        if len(x) > 2:
            return (x[:-1], y)
        return (x[0], y)

    @staticmethod
    def get_text(sample: Tuple[Any, Any]) -> List[str]:
        x, y = sample
        if not isinstance(x[0], str):
            return x[0]
        return x 

    @staticmethod
    def merge_samples(samples: List[Tuple[Any, Any]]) -> Tuple[Any, Any]:
        out_x, out_y = [], []
        for x, y in samples:
            if not isinstance(x[0], str):
                if not out_x:
                    out_x = [[]] * len(x)
                out_x = tuple(out_x_i + x_i for out_x_i, x_i in zip(out_x, x))
            else:
                out_x.extend(x)
            out_y.extend(y)
        return (out_x, out_y)

    def sample_from_doc(self, doc: List[Tuple[Any, Any]]) -> Tuple[Any, Any]:
        sample_id = self.random.randint(0, len(doc) - 1)
        doc_texts = [self.get_text(s) for s in doc]
        rich_sample_ids = self.get_context_indices(doc_texts,
                                                   sample_id=sample_id,
                                                   subtokenizer=self.tokenizer,
                                                   max_subtokens_length=self.max_seq_length,
                                                   left_context_rate=self.left_context_rate,
                                                   random=self.random)
        return self.merge_samples((doc[i] for i in rich_sample_ids))

    def chunks_from_doc(self, doc: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        pull_of_samples = copy(doc)
        pull_of_texts = [self.get_text(s) for s in doc]
        while pull_of_samples:
            rich_sample_ids = self.get_context_indices(pull_of_texts,
                                                       sample_id=0,
                                                       subtokenizer=self.tokenizer,
                                                       max_subtokens_length=self.max_seq_length,
                                                       left_context_rate=0.0,
                                                       random=self.random)
            # TODO: split differently & replace tags with 'X's for contexts
            yield self.merge_samples((pull_of_samples[i] for i in rich_sample_ids))
            pull_of_samples = pull_of_samples[len(rich_sample_ids):]
            pull_of_texts = pull_of_texts[len(rich_sample_ids):]
            if len(rich_sample_ids) != max(rich_sample_ids) + 1:
                raise RuntimeError("can't split doc {doc} into chunks")

    @staticmethod
    def get_context_indices(samples: List[List[str]],
                            sample_id: int,
                            subtokenizer: FullTokenizer,
                            max_subtokens_length: int,
                            left_context_rate: float = 0.5,
                            random: Random = Random(31)) -> List[int]:
        rich_sample_indices = [sample_id]

        toks = samples[sample_id]
        l_ctx = samples[:sample_id]
        r_ctx = samples[sample_id + 1:]

        subtoks_len = len([st for t in toks
                           for st in subtokenizer.tokenize(t)])
        l_i, r_i = 0, 0
        while (l_i < len(l_ctx)) or (r_i < len(r_ctx)):
            l_rate = left_context_rate if r_i < len(r_ctx) else 1.0
            if (l_i < len(l_ctx)) and (random.random() < l_rate):
                # add one sentence from left_context
                subtoks = [st for t in l_ctx[-l_i-1]
                           for st in subtokenizer.tokenize(t)]
                if subtoks_len + len(subtoks) > max_subtokens_length:
                    break
                subtoks_len += len(subtoks)
                rich_sample_indices = [sample_id - l_i - 1] + rich_sample_indices 
                l_i += 1
            else:
                # add one sentence from right_context
                subtoks = [st for t in r_ctx[r_i] for st in subtokenizer.tokenize(t)]
                if subtoks_len + len(subtoks) > max_subtokens_length:
                    break
                subtoks_len += len(subtoks)
                rich_sample_indices.append(sample_id + r_i + 1)
                r_i += 1
        return rich_sample_indices

