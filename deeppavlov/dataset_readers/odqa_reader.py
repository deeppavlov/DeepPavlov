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
import logging
import sqlite3
import unicodedata
from multiprocessing import Pool
from pathlib import Path
from typing import Union, List, Tuple, Generator, Any, Optional

from tqdm import tqdm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download

logger = logging.getLogger(__name__)


@register('odqa_reader')
class ODQADataReader(DatasetReader):
    """Build a SQLite database from folder with txt files, json files or
    `Wiki Extractor <https://github.com/attardi/wikiextractor>`_ files.

    """

    def read(self, data_path: Union[Path, str], db_url: Optional[str] = None, *args,
             **kwargs) -> None:
        """Build a SQLite database from provided files, download SQLite database from a provided URL,
         or do nothing.

        Args:
            data_path: a directory/file with texts to create a database from
            db_url: path to a database url
            kwargs:
                save_path: a path where a database should be saved to, or path to a ready database
                dataset_format: initial data format; should be selected from ['txt', 'wiki', 'json']

        Returns:
            None

        """
        logger.info('Reading files...')
        try:
            save_path = expand_path(kwargs['save_path'])
        except KeyError:
            raise ConfigError(
                f'\"save_path\" attribute should be set for {self.__class__.__name__}\
                 in the JSON config.')
        if save_path.exists() and save_path.with_suffix(f'{save_path.suffix}.done').exists():
            return
        try:
            dataset_format = kwargs['dataset_format']
        except KeyError:
            raise ConfigError(
                f'\"dataset_format\" attribute should be set for {self.__class__.__name__}\
                 in the JSON config.')

        save_path.parent.mkdir(parents=True, exist_ok=True)

        if db_url:
            download_dir = save_path.parent
            logger.info(f'Downloading database from {db_url} to {download_dir}')
            download(download_dir, db_url, force_download=False)
            return

        self._build_db(save_path, dataset_format, expand_path(data_path))

    def iter_files(self, path: Union[Path, str]) -> Generator[Path, Any, Any]:
        """Iterate over folder with files or a single file and generate file paths.

        Args:
            path: path to a folder or a file

        Raises:
            RuntimeError if the provided `path` doesn't exist

        Yields:
            file paths one by one

        Returns:
            None

        """
        path = Path(path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for item in path.iterdir():
                yield from self.iter_files(item)
        else:
            raise RuntimeError("Path doesn't exist: {}".format(path))

    def _build_db(self, save_path: Union[Path, str], dataset_format: str,
                  data_path: Union[Path, str],
                  num_workers: int = 8) -> None:
        """Build a SQLite database in parallel and save it to a pointed path.

        Args:
            save_path: a path where the ready database should be saved
            dataset_format: a data format, should be selected from ['txt', 'json', 'wiki']
            data_path: path to a folder/file from which to build a database
            num_workers: a number of workers for parallel database building

        Raises:
            sqlite3.OperationalError if `save_path` doesn't exist.
            RuntimeError if dataset_format is not in ['txt', 'json', 'wiki']

        Returns:
            None

        """
        done_path = save_path.with_suffix(f'{save_path.suffix}.done')

        if Path(save_path).exists():
            Path(save_path).unlink()
        if done_path.exists():
            done_path.unlink()

        logger.info('Building the database...')

        try:
            conn = sqlite3.connect(str(save_path))
        except sqlite3.OperationalError as e:
            e.args = e.args + ("Check that DB path exists.",)
            raise e
        c = conn.cursor()
        sql_table = "CREATE TABLE documents (id PRIMARY KEY, text);"
        c.execute(sql_table)

        files = [f for f in self.iter_files(data_path)]
        workers = Pool(num_workers)

        if dataset_format == 'txt':
            fn = self._get_file_contents
        elif dataset_format == 'json':
            fn = self._get_json_contents
        elif dataset_format == 'wiki':
            fn = self._get_wiki_contents
        else:
            raise RuntimeError('Unknown dataset format.')

        with tqdm(total=len(files)) as pbar:
            for data in tqdm(workers.imap_unordered(fn, files)):
                try:
                    c.executemany("INSERT INTO documents VALUES (?,?)", data)
                    pbar.update()
                except sqlite3.IntegrityError as e:
                    logger.warning(e)

        conn.commit()
        conn.close()
        done_path.touch()

    @staticmethod
    def _get_file_contents(fpath: Union[Path, str]) -> List[Tuple[str, str]]:
        """Extract file contents from '.txt' file.

        Args:
            fpath: path to a '.txt' file.

        Returns:
             a list with tuple of normalized file name and file contents

        """
        with open(fpath, encoding='utf-8') as fin:
            text = fin.read()
            normalized_text = unicodedata.normalize('NFD', text)
            return [(fpath.name, normalized_text)]

    @staticmethod
    def _get_json_contents(fpath: Union[Path, str]) -> List[Tuple[str, str]]:
        """Extract file contents from '.json' file. JSON files should be formatted as list with dicts
        which contain 'title' and 'doc' keywords.

        Args:
            fpath: path to a '.json' file.

        Returns:
            a list with tuples of normalized file name and file contents

        """
        docs = []
        with open(fpath, encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line)
                for doc in data:
                    if not doc:
                        continue
                    text = doc['text']
                    normalized_text = unicodedata.normalize('NFD', text)
                    docs.append((doc['title'], normalized_text))
        return docs

    @staticmethod
    def _get_wiki_contents(fpath: Union[Path, str]) -> List[Tuple[str, str]]:
        """Extract file contents from wiki extractor formatted files.

        Args:
            fpath: path to a '.txt' file in wiki extractor format

        Returns:
            a list with tuples of normalized file name and file contents

        """
        docs = []
        with open(fpath, encoding='utf-8') as fin:
            for line in fin:
                doc = json.loads(line)
                if not doc:
                    continue
                text = doc['text']
                normalized_text = unicodedata.normalize('NFD', text)
                docs.append((doc['title'], normalized_text))
        return docs
