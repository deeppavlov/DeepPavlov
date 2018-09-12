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
from typing import List, Any, Dict, Optional
from random import Random

from overrides import overrides

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator

logger = get_logger(__name__)

DB_URL = 'http://files.deeppavlov.ai/datasets/wikipedia/enwiki.db'


@register('sqlite_iterator')
class SQLiteDataIterator(DataFittingIterator):
    """Iterate over SQLite database.
    Gen batches from SQLite data.
    Get document ids and document.

    Args:
        data_dir: a directory where to save downloaded DB to
        data_url: an URL where to download a DB from
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

    def __init__(self, data_dir: str = '', data_url: str = DB_URL, batch_size: int = None,
                 shuffle: bool = None, seed: int = None, **kwargs):

        download_dir = expand_path(data_dir)
        download_path = download_dir.joinpath(data_url.split("/")[-1])
        download(download_path, data_url, force_download=False)

        self.connect = sqlite3.connect(str(download_path), check_same_thread=False)
        self.db_name = self.get_db_name()
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
