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

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download, mark_done

logger = logging.getLogger(__name__)


@register('classification_datasetreader')
class ClassificationDatasetReader(DatasetReader):
    """
    Class provides reading dataset in .csv format
    """

    url = 'http://lnsigo.mipt.ru/export/datasets/snips_intents/train.csv'

    @overrides
    def read(self, data_path, data_types=["train"]):
        """
        Read dataset from data_path directory.
        Reading files are all data_types + extension
        (i.e for data_types=["train", "valid"] files "train.csv" and "valid.csv" form
        data_path will be read)
        Args:
            data_path: directory with files
            data_types: types of considered data (possible: "train", "valid", "test")

        Returns:
            dictionary with types from data_types.
            Each field of dictionary is a list of tuples (x_i, y_i)
        """

        for data_type in data_types:
            if not Path(data_path).joinpath(data_type + ".csv").exists():
                print("Loading {} data from {} to {}".format(data_type, self.url, data_path))
                download(source_url=self.url,
                         dest_file_path=Path(data_path).joinpath(data_type + ".csv"))
                mark_done(data_path)

        data = {}
        for data_type in data_types:
            data[data_type] = pd.read_csv(Path(data_path).joinpath(data_type + ".csv"))

        new_data = {'train': [],
                    'valid': [],
                    'test': []}
        columns = np.array(data["train"].columns)

        for field in data_types:
            for i in range(data[field].shape[0]):
                new_data[field].append(
                    (data[field].loc[i, 'text'], list(columns[data[field].loc[i, columns] == 1.0])))

        return new_data
