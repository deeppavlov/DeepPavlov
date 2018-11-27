from typing import Dict, List, Tuple
import json

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register
from deeppavlov.core.commands.utils import expand_path


@register("yandex_reader")
class YandexReader(DatasetReader):

    def read(self,
             data_path: str,
             seed: int = None, *args, **kwargs) -> Dict[str, List[Tuple[List[str], int]]]:
        data_path = expand_path(data_path)
        train_fname = data_path / 'jaccard.json'
        test_fname = data_path / 'yandex_test.json'
        train_data = self.build_data(train_fname)
        test_data = self.build_data(test_fname)
        dataset = {"train": train_data, "valid": [], "test": test_data}
        return dataset


    def int_class(self, str_y):
        if str_y == '-1':
            return 0
        else:
            return 1

    def build_data(self, name):
        with open(name) as f:
            data = json.load(f)
        return [([doc['text_1'], doc['text_2']], self.int_class(doc['class'])) for doc in data]


