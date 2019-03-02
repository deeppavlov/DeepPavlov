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

import sqlite3
from logging import getLogger
from pathlib import Path
from random import Random
from typing import List, Any, Dict, Optional, Union, Generator, Tuple

from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator

logger = getLogger(__name__)


@register('sqlite_iterator')
class SQLiteDataIterator(DataFittingIterator):
    """Iterate over SQLite database.
    Gen batches from SQLite data.
    Get document ids and document.

    Args:
        load_path: a path to local DB file
        batch_size: a number of samples in a single batch
        shuffle: whether to shuffle data during batching
        seed: random seed for data shuffling

    Attributes:
        connect: a DB connection
        db_name: a DB name
        doc_ids: DB document ids
        doc2index: a dictionary of document indices and their titles
        batch_size: a number of samples in a single batch
        shuffle: whether to shuffle data during batching
        random: an instance of :class:`Random` class.

    """

    def __init__(self, load_path: Union[str, Path], batch_size: Optional[int] = None,
                 shuffle: Optional[bool] = None, seed: Optional[int] = None, **kwargs) -> None:

        load_path = str(expand_path(load_path))
        logger.info("Connecting to database, path: {}".format(load_path))
        try:
            self.connect = sqlite3.connect(load_path, check_same_thread=False)
        except sqlite3.OperationalError as e:
            e.args = e.args + ("Check that DB path exists and is a valid DB file",)
            raise e
        try:
            self.db_name = self.get_db_name()
        except TypeError as e:
            e.args = e.args + (
                'Check that DB path was created correctly and is not empty. '
                'Check that a correct dataset_format is passed to the ODQAReader config',)
            raise e
        self.doc_ids = self.get_doc_ids()
        self.doc2index = self.map_doc2idx()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = Random(seed)

    @overrides
    def get_doc_ids(self) -> List[Any]:
        """Get document ids.

        Returns:
            document ids
        """
        cursor = self.connect.cursor()
        cursor.execute('SELECT id FROM {}'.format(self.db_name))
        ids = [ids[0] for ids in cursor.fetchall()]
        cursor.close()
        return ids

    def get_db_name(self) -> str:
        """Get DB name.

        Returns:
            DB name

        """
        cursor = self.connect.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        assert cursor.arraysize == 1
        name = cursor.fetchone()[0]
        cursor.close()
        return name

    def map_doc2idx(self) -> Dict[int, Any]:
        """Map DB ids to integer ids.

        Returns:
            a dictionary of document titles and correspondent integer indices

        """
        doc2idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        logger.info(
            "SQLite iterator: The size of the database is {} documents".format(len(doc2idx)))
        return doc2idx

    @overrides
    def get_doc_content(self, doc_id: Any) -> Optional[str]:
        """Get document content by id.

        Args:
            doc_id: a document id

        Returns:
            document content if success, else raise Exception

        """
        cursor = self.connect.cursor()
        cursor.execute(
            "SELECT text FROM {} WHERE id = ?".format(self.db_name),
            (doc_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]

    @overrides
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

        for i, doc_ids in enumerate(batches):
            docs = [self.get_doc_content(doc_id) for doc_id in doc_ids]
            doc_nums = [self.doc2index[doc_id] for doc_id in doc_ids]
            yield docs, zip(doc_ids, doc_nums)

    def get_instances(self):
        """Get all data"""
        doc_ids = list(self.doc_ids)
        docs = [self.get_doc_content(doc_id) for doc_id in doc_ids]
        doc_nums = [self.doc2index[doc_id] for doc_id in doc_ids]
        return docs, zip(doc_ids, doc_nums)
