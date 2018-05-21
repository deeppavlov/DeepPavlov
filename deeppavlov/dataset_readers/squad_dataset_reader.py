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

from pathlib import Path
import json

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register


@register('squad_dataset_reader')
class SquadDatasetReader(DatasetReader):
    """
    Stanford Question Answering Dataset
    https://rajpurkar.github.io/SQuAD-explorer/
    and
    Russian dataset from SDSJ
    https://www.sdsj.ru/ru/contest.html
    """

    url_squad = 'http://lnsigo.mipt.ru/export/datasets/squad-v1.1.tar.gz'
    url_sber_squad = 'http://lnsigo.mipt.ru/export/datasets/sber_squad-v1.1.tar.gz'

    def read(self, dir_path: str, dataset='SQuAD'):
        if dataset == 'SQuAD':
            self.url = self.url_squad
        elif dataset == 'SberSQuAD':
            self.url = self.url_sber_squad
        else:
            raise RuntimeError('Dataset {} is unknown'.format(dataset))

        dir_path = Path(dir_path)
        required_files = ['{}-v1.1.json'.format(dt) for dt in ['train', 'dev']]
        if not dir_path.exists():
            dir_path.mkdir()

        if not all((dir_path / f).exists() for f in required_files):
            download_decompress(self.url, dir_path)

        dataset = {}
        for f in required_files:
            data = json.load((dir_path / f).open('r'))
            if f == 'dev-v1.1.json':
                dataset['valid'] = data
            else:
                dataset['train'] = data

        return dataset
