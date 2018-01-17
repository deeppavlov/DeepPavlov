"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, softwaredata
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
from itertools import chain
from pathlib import Path

import pandas as pd
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download, mark_done

logger = logging.getLogger(__name__)


@register('classification_datasetreader')
class ClassificationDatasetReader(DatasetReader):

    url = 'http://lnsigo.mipt.ru/export/datasets/snips_intent_data.csv'

    @overrides
    def read(self, data_path, data_types=["train"]):

        for data_type in data_types:
            if not Path(data_path).joinpath(data_type + ".csv").exists():
                print("Loading {} data from {} to {}".format(data_type, self.url, data_path))
                download(self.url, data_path)
                mark_done(data_path)

        data = {}
        for data_type in data_types:
            data[data_type] = pd.read_csv(Path(data_path).joinpath(data_type + ".csv"))

        return data

