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
import pickle

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.registry import register


@register('ontonotes_reader')
class OntonotesReader(DatasetReader):
    URL = 'http://lnsigo.mipt.ru/export/datasets/ontonotes_senna.pckl'

    def read(self, data_path, file_name: str='ontonotes_senna.pckl', provide_pos=True):
        path = Path(data_path).resolve() / file_name
        if not path.exists():
            download_decompress(self.URL, str(path.parent))
        with open(path, 'rb') as f:
            return pickle.load(f)
