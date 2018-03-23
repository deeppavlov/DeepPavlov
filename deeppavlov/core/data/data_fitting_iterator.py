"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from random import Random
from typing import List, Generator, Tuple, Any

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


@register('data_fitting_iterator')
class DataFittingIterator:
    """
    Dataset iterator for fitting models, e. g. vocabs, kNN, vectorizers.
    Data is passed as a list of strings(documents).
    Generate batches (for large datasets).
    """

    def __init__(self, data: List[str], doc_ids: List[Any]=None,
                 seed: int = None, shuffle: bool = None, batch_size: int = None,
                 *args, **kwargs) -> None:

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = Random(seed)
        self.data = data
        self.doc_index = doc_ids or self.get_doc_ids()
        self.data = data

    def get_doc_ids(self):
        ids = [i for i in range(len(self.data))]
        return ids

    def gen_batch(self, batch_size=1000, shuffle=False) -> Generator[Tuple[List[list],
                                                                           List[int]], Any, None]:
        _batch_size = self.batch_size or batch_size
        _shuffle = self.shuffle or shuffle

        if _shuffle:
            self.random.shuffle(self.doc_index)

        batches = [self.doc_index[i:i + _batch_size] for i in
                   range(0, len(self.doc_index), _batch_size)]

        # DEBUG
        # len_batches = len(batches)

        for i, doc_ids in enumerate(batches):
            # DEBUG
            # logger.info(
            #     "Processing batch # {} of {} ({} documents)".format(i, len_batches, len(doc_ids)))
            docs = [self.data[i] for i in doc_ids]
            yield docs, doc_ids
