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

import csv
import requests
from pathlib import Path

from lxml import html

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import is_done, download, mark_done
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('typos_custom_reader')
class TyposCustom(DatasetReader):
    @staticmethod
    def save_vocab(data, ser_dir):
        pass

    def __init__(self):
        pass

    @staticmethod
    def build(data_path: str):
        return Path(data_path)

    @classmethod
    def read(cls, data_path: str, *args, **kwargs):
        fname = cls.build(data_path)
        with fname.open(newline='', encoding='utf8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader)
            res = [(mistake, correct) for mistake, correct in reader]
        return {'train': res}


@register('typos_wikipedia_reader')
class TyposWikipedia(TyposCustom):
    @staticmethod
    def build(data_path: str):
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
    def __init__(self):
        pass

    @staticmethod
    def build(data_path: str):
        data_path = Path(data_path) / 'kartaslov'

        fname = data_path / 'orfo_and_typos.L1_5.csv'

        if not is_done(data_path):
            url = 'https://raw.githubusercontent.com/dkulagin/kartaslov/master/dataset/orfo_and_typos/orfo_and_typos.L1_5.csv'

            download(fname, url)

            mark_done(data_path)

            log.info('Built')
        return fname

    @staticmethod
    def read(data_path: str, *args, **kwargs):
        fname = TyposKartaslov.build(data_path)
        with open(str(fname), newline='', encoding='utf8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader)
            res = [(mistake, correct) for correct, mistake, weight in reader]
        return {'train': res}
