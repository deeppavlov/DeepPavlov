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

import csv
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from lxml import html

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import is_done, download, mark_done

log = getLogger(__name__)


@register('typos_custom_reader')
class TyposCustom(DatasetReader):
    """Base class for reading spelling corrections dataset files

    """

    def __init__(self):
        pass

    @staticmethod
    def build(data_path: str) -> Path:
        """Base method that interprets ``data_path`` argument.

        Args:
            data_path: path to the tsv-file containing erroneous and corrected words

        Returns:
            the same path as a :class:`~pathlib.Path` object
        """
        return Path(data_path)

    @classmethod
    def read(cls, data_path: str, *args, **kwargs) -> Dict[str, List[Tuple[str, str]]]:
        """Read train data for spelling corrections algorithms

        Args:
            data_path: path that needs to be interpreted with :meth:`~deeppavlov.dataset_readers.typos_reader.TyposCustom.build`

        Returns:
            train data to pass to a :class:`~deeppavlov.dataset_iterators.typos_iterator.TyposDatasetIterator`
        """
        fname = cls.build(data_path)
        with fname.open(newline='', encoding='utf8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader)
            res = [(mistake, correct) for mistake, correct in reader]
        return {'train': res}


@register('typos_wikipedia_reader')
class TyposWikipedia(TyposCustom):
    """Implementation of :class:`~deeppavlov.dataset_readers.typos_reader.TyposCustom` that works with
     English Wikipedia's list of common misspellings

    """

    @staticmethod
    def build(data_path: str) -> Path:
        """Download and parse common misspellings list from `Wikipedia <https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines>`_

        Args:
            data_path: target directory to download the data to

        Returns:
            path to the resulting tsv-file
        """
        data_path = Path(data_path) / 'typos_wiki'

        fname = data_path / 'misspelings.tsv'

        if not is_done(data_path):
            url = 'https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines'

            page = requests.get(url)
            tree = html.fromstring(page.content)
            raw = tree.xpath('//pre/text()')[0].splitlines()
            data = []
            for pair in raw:
                typo, corrects = pair.strip().split('->')
                for correct in corrects.split(','):
                    data.append([typo.strip(), correct.strip()])

            fname.parent.mkdir(parents=True, exist_ok=True)
            with fname.open('w', newline='', encoding='utf8') as tsvfile:
                writer = csv.writer(tsvfile, delimiter='\t')
                for line in data:
                    writer.writerow(line)

            mark_done(data_path)

            log.info('Built')
        return fname


@register('typos_kartaslov_reader')
class TyposKartaslov(DatasetReader):
    """Implementation of :class:`~deeppavlov.dataset_readers.typos_reader.TyposCustom` that works with
     a Russian misspellings dataset from `kartaslov <https://github.com/dkulagin/kartaslov>`_

    """

    def __init__(self):
        pass

    @staticmethod
    def build(data_path: str) -> Path:
        """Download misspellings list from `github <https://raw.githubusercontent.com/dkulagin/kartaslov/master/dataset/orfo_and_typos/orfo_and_typos.L1_5.csv>`_

        Args:
            data_path: target directory to download the data to

        Returns:
            path to the resulting csv-file
        """
        data_path = Path(data_path) / 'kartaslov'

        fname = data_path / 'orfo_and_typos.L1_5.csv'

        if not is_done(data_path):
            url = 'https://raw.githubusercontent.com/dkulagin/kartaslov/master/dataset/orfo_and_typos/orfo_and_typos.L1_5.csv'

            download(fname, url)

            mark_done(data_path)

            log.info('Built')
        return fname

    @staticmethod
    def read(data_path: str, *args, **kwargs) -> Dict[str, List[Tuple[str, str]]]:
        """Read train data for spelling corrections algorithms

        Args:
            data_path: path that needs to be interpreted with :meth:`~deeppavlov.dataset_readers.typos_reader.TyposKartaslov.build`

        Returns:
            train data to pass to a :class:`~deeppavlov.dataset_iterators.typos_iterator.TyposDatasetIterator`
        """
        fname = TyposKartaslov.build(data_path)
        with open(str(fname), newline='', encoding='utf8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader)
            res = [(mistake, correct) for correct, mistake, weight in reader]
        return {'train': res}
