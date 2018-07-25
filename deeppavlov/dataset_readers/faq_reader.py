from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register

from pandas import read_csv
from numpy import nan
from typing import Dict

@register('faq_reader')
class FaqDatasetReader(DatasetReader):
    

    def read(self, dataset_path: str, *args, **kwargs) -> Dict:
        """
        Read dataset from specified csv: dataset_path.
        Args:
            dataset_path: csv file with faq and paraphrases

        Returns:
            dict dataset["train"]
        """
        data = read_csv(dataset_path)
        cols_names = [c for c in data.columns.values if c not in ['Question', 'Answer']]

        x_phrases = []
        y_phrases = []
        for idx, row in data.iterrows():
            for col in cols_names:
                if row[col] is not nan:
                    x_phrases.append(row[col])
                    y_phrases.append(row['Answer'])

        x_questions = data['Question'].values.tolist()
        y_questions = data['Answer'].values.tolist()

        train_xy_tuples = [(x_questions[i], y_questions[i]) for i in range(len(x_questions))]
        train_xy_tuples = train_xy_tuples + [(x_phrases[i], y_phrases[i]) for i in range(len(x_phrases))]

        dataset = {'train': None, 'valid': None, 'test': None}
        dataset["train"] = train_xy_tuples
        dataset["valid"] = []
        dataset["test"] = []

        return dataset

