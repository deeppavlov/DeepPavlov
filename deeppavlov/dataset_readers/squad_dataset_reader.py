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


from typing import Dict, Any
from pathlib import Path
import json

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register


@register('squad_dataset_reader')
class SquadDatasetReader(DatasetReader):
    """
    Downloads dataset files and prepares train/valid split.

    SQuAD:
    Stanford Question Answering Dataset
    https://rajpurkar.github.io/SQuAD-explorer/

    SberSQuAD:
    Dataset from SDSJ Task B
    https://www.sdsj.ru/ru/contest.html

    MultiSQuAD:
    SQuAD dataset with additional contexts retrieved (by tfidf) from original Wikipedia article.
    """

    url_squad = 'http://files.deeppavlov.ai/datasets/squad-v1.1.tar.gz'
    url_sber_squad = 'http://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz'
    url_multi_squad = 'http://files.deeppavlov.ai/datasets/multiparagraph_squad.tar.gz'

    def read(self, dir_path: str, dataset: str = 'SQuAD', *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        """

        Args:
            dir_path: path to save data
            dataset: dataset name: ``'SQuAD'``, ``'SberSQuAD'`` or ``'MultiSQuAD'``

        Returns:
            dataset split on train/valid

        Raises:
            RuntimeError: if `dataset` is not one of these: ``'SQuAD'``, ``'SberSQuAD'``, ``'MultiSQuAD'``.
        """
        if dataset == 'SQuAD':
            self.url = self.url_squad
        elif dataset == 'SberSQuAD':
            self.url = self.url_sber_squad
        elif dataset == 'MultiSQuAD':
            self.url = self.url_multi_squad
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
            with dir_path.joinpath(f).open('r', encoding='utf8') as fp:
                data = json.load(fp)
            if f == 'dev-v1.1.json':
                dataset['valid'] = data
            else:
                dataset['train'] = data

        return dataset
