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

import pandas as pd
from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download, mark_done
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('classification_datasetreader')
class ClassificationDatasetReader(DatasetReader):
    """
    Class provides reading dataset in .csv format
    """

    url = 'http://lnsigo.mipt.ru/export/datasets/snips_intents/train.csv'

    @overrides
    def read(self, data_path):
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
        data_types = ["train", "valid", "test"]

        if not Path(data_path, "train.csv").exists():
            log.info("Loading train data from {} to {}".format(self.url, data_path))
            download(source_url=self.url, dest_file_path=Path(data_path, "train.csv"))

        data = {"train": [],
                "valid": [],
                "test": []}
        for data_type in data_types:
            try:
                df = pd.read_csv(Path(data_path).joinpath(data_type + ".csv"))
                data[data_type] = [(row['text'], row['intents'].split(',')) for _, row in df.iterrows()]
            except FileNotFoundError:
                log.warning("Cannot find {}.csv data file".format(data_type))

        return data
