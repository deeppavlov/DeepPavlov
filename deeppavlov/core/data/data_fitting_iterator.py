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

from logging import getLogger
from random import Random
from typing import List, Generator, Tuple, Any, Optional

from deeppavlov.core.common.registry import register

logger = getLogger(__name__)


@register('data_fitting_iterator')
class DataFittingIterator:
    """Dataset iterator for fitting estimator models, like vocabs, kNN, vectorizers.
    Data is passed as a list of strings(documents).
    Generate batches (for large datasets).

    Args:
        data: list of documents
        doc_ids: provided document ids
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching

    Attributes:
        shuffle: whether to shuffle data during batching
        random: instance of :class:`Random` initialized with a seed
        data: list of documents
        doc_ids: provided by a user ids or generated automatically ids

    """

    def __init__(self, data: List[str], doc_ids: List[Any] = None,
                 seed: int = None, shuffle: bool = True,
                 *args, **kwargs) -> None:

        self.shuffle = shuffle
        self.random = Random(seed)
        self.data = data
        self.doc_ids = doc_ids or self.get_doc_ids()

    def get_doc_ids(self):
        """Generate doc ids.

        Returns: doc ids

        """
        return list(range(len(self.data)))

    def get_doc_content(self, doc_id: Any) -> Optional[str]:
        """Get doc content by id.

        Args:
            doc_id: an id for a doc which content should be extracted

        Returns:
            doc content as a string if id exists or raise an error

        """
        return self.data[doc_id]

    def gen_batches(self, batch_size: int, shuffle: bool = None) \
            -> Generator[Tuple[List[str], List[int]], Any, None]:
        """Gen batches of documents.

        Args:
            batch_size: a number of samples in a single batch
            shuffle: whether to shuffle data during batching

        Yields:
            generated tuple of documents and their ids

        """
        if shuffle is None:
            shuffle = self.shuffle

        if shuffle:
            _doc_ids = self.random.sample(self.doc_ids, len(self.doc_ids))
        else:
            _doc_ids = self.doc_ids

        if batch_size > 0:
            batches = [_doc_ids[i:i + batch_size] for i in
                       range(0, len(_doc_ids), batch_size)]
        else:
            batches = [_doc_ids]

        # DEBUG
        # len_batches = len(batches)

        for i, doc_ids in enumerate(batches):
            # DEBUG
            # logger.info(
            #     "Processing batch # {} of {} ({} documents)".format(i, len_batches, len(doc_index)))
            docs = [self.get_doc_content(doc_id) for doc_id in doc_ids]
            yield docs, doc_ids

    def get_instances(self):
        """Get all data"""
        doc_ids = list(self.doc_ids)
        docs = [self.get_doc_content(doc_id) for doc_id in doc_ids]
        return docs, doc_ids
