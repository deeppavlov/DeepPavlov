import random
from pathlib import Path

import numpy as np
from typing import List, Dict, Generator, Tuple, Any
from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset
from deeppavlov.core.common import paths
# from deeppavlov.models.embedders.fasttext_embedder import EmbeddingsDict
# from deeppavlov.models.intent_recognition.intent_keras.intent_model import KerasIntentModel
# from deeppavlov.models.intent_recognition.intent_keras.utils import labels2onehot, proba2labels, proba2onehot


@register('intent_dataset')
class IntentDataset(Dataset):

    def __init__(self, data, dataset_dir='intents', dataset_file='classes.txt',
                 seed=None, extract_classes=True, classes_file=None,
                 fields_to_merge=None, merged_field=None,
                 field_to_split=None, splitted_fields=None, splitting_proportions=None,
                 *args, **kwargs):

        super().__init__(data, seed)
        self.classes = None

        if extract_classes:
            self.classes = self._extract_classes()
            if classes_file is None:
                # mkdir dataseT_dir
                ser_dir = Path(paths.USR_PATH).joinpath(dataset_dir)
                if not ser_dir.exists():
                    ser_dir.mkdir()
                classes_file = Path(paths.USR_PATH).joinpath(dataset_dir, dataset_file)
                print("No file name for classes provided. Classes are saved to file {}".format(classes_file))
            with open(Path(classes_file), 'w') as fin:
                for i in range(len(self.classes)):
                    fin.write(self.classes[i] + '\n')
        if fields_to_merge is not None:
            if merged_field is not None:
                print("Merging fields <<{}>> to new field <<{}>>".format(fields_to_merge, merged_field))
                self._merge_data(fields_to_merge=fields_to_merge.split(' '), merged_field=merged_field)
            else:
                raise IOError("Given fields to merge BUT not given name of merged field")

        if field_to_split is not None:
            if splitted_fields is not None:
                print("Splitting field <<{}>> to new fields <<{}>>".format(field_to_split, splitted_fields))
                self._split_data(field_to_split=field_to_split,
                                 splitted_fields=splitted_fields.split(" "),
                                 splitting_proportions=[float(s) for s in splitting_proportions.split(" ")])
            else:
                raise IOError("Given field to split BUT not given names of splitted fields")

    def _extract_classes(self, *args, **kwargs):
        intents = []
        all_data = self.iter_all(data_type='train')
        for sample in all_data:
            intents.extend(sample[1])
        if 'valid' in self.data.keys():
            all_data = self.iter_all(data_type='valid')
            for sample in all_data:
                intents.extend(sample[1])
        intents = np.unique(intents)
        return np.array(sorted(intents))

    def _split_data(self, field_to_split, splitted_fields, splitting_proportions):
        data_to_div = self.data[field_to_split].copy()
        data_size = len(self.data[field_to_split])
        for i in range(len(splitted_fields) - 1):
            self.data[splitted_fields[i]], data_to_div = train_test_split(data_to_div,
                                                                          test_size=
                                                                          len(data_to_div) -
                                                                          int(data_size * splitting_proportions[i]))
        self.data[splitted_fields[-1]] = data_to_div
        return True

    def _merge_data(self, fields_to_merge, merged_field):
        data = self.data.copy()
        data[merged_field] = []
        for name in fields_to_merge:
            data[merged_field] += self.data[name]
        self.data = data
        return True
