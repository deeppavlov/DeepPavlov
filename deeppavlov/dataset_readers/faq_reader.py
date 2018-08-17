from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register

from pandas import read_csv
from typing import Dict

@register('faq_reader')
class FaqDatasetReader(DatasetReader):
    

    def read(self, data_path: str=None, data_url: str=None, x_col_name: str='x', y_col_name: str='y', *args, **kwargs) -> Dict:
        """
        Read dataset from specified csv: data_path or data_url.
        Args:
            data_path: path to csv file with faq and paraphrases
            data_url: url of csv file with faq and paraphrases
        Returns:
            dict dataset["train"]
        """

        if len(data_url)!=0:
            data = read_csv(data_url)
        else:
            data = read_csv(data_path)

        X = data[x_col_name]
        y = data[y_col_name]

        train_xy_tuples = [(X[i], y[i]) for i in range(len(X))]

        dataset = {'train': None, 'valid': None, 'test': None}
        dataset["train"] = train_xy_tuples
        dataset["valid"] = []
        dataset["test"] = []


        return dataset

