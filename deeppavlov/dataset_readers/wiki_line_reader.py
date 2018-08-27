from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register

from pandas import read_csv


@register('wiki_line_reader')
class FaqDatasetReader(DatasetReader):
    
    def read(self, data_path: str=None, *args, **kwargs):

        with open(data_path) as f:
            content = f.readlines()

        dataset = dict()
        dataset["train"] = [(line, ) for line in content]
        dataset["valid"] = []
        dataset["test"] = []

        return dataset
