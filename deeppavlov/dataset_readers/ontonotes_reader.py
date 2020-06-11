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

import pickle
from logging import getLogger
from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download

log = getLogger(__name__)


@register('ontonotes_reader')
class OntonotesReader(DatasetReader):
    """Class to read training datasets in OntoNotes format"""
    URL = 'http://files.deeppavlov.ai/datasets/ontonotes_senna.pckl'

    def __init__(self):
        log.warning('ontonotes_reader is deprecated and will be removed in future versions.'
                    ' Please, use conll2003_reader with `"dataset_name": "ontonotes"` instead')

    def read(self, data_path, file_name: str = 'ontonotes_senna.pckl', provide_senna_pos=False,
             provide_senna_ner=False):
        path = Path(data_path).resolve() / file_name
        if not path.exists():
            download(str(path), self.URL)
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        dataset_filtered = {}
        for key, data in dataset.items():
            dataset_filtered[key] = []
            for (toks, pos, ner), tags in data:
                if not provide_senna_pos and not provide_senna_ner:
                    dataset_filtered[key].append((toks, tags))
                else:
                    x = [toks]
                    if provide_senna_pos:
                        x.append(pos)
                    if provide_senna_ner:
                        x.append(ner)
                    dataset_filtered[key].append((x, tags))

        return dataset_filtered
