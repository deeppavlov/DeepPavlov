from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register

from pandas import read_csv


@register('faq_reader')
class FaqDatasetReader(DatasetReader):
    
    def read(self, data_path: str=None, data_url: str=None, x_col_name: str='x', y_col_name: str='y', *args, **kwargs):

        if len(data_url) != 0:
            data = read_csv(data_url)
        else:
            data = read_csv(data_path)

        x = data[x_col_name]
        y = data[y_col_name]

        train_xy_tuples = [(x[i], y[i]) for i in range(len(x))]

        dataset = dict()
        dataset["train"] = train_xy_tuples
        dataset["valid"] = []
        dataset["test"] = []

        return dataset
