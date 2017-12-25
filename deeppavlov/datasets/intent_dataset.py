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

import random
from typing import List, Dict, Generator, Tuple, Any
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset import Dataset
from deeppavlov.models.embedders.fasttext_embedder import EmbeddingsDict
from deeppavlov.models.intent_recognition.intent_keras.intent_model import KerasIntentModel
from deeppavlov.models.intent_recognition.intent_keras.utils import labels2onehot, proba2labels, proba2onehot


@register('intent_dataset')
class IntentDataset(Dataset):

    def __init__(self, data, seed=None, extract_classes=True, classes_file=None,
                 fields_to_merge=None, merged_field=None,
                 field_to_split=None, splitted_fields=None, splitting_proportions=None,
                 *args, **kwargs):

        super().__init__(data, seed)
        self.classes = None

        if extract_classes == True:
            self.classes = self._extract_classes()
            if classes_file is None:
                classes_file = "./classes.txt"
                print("No file name for classes provided. Classes are saved to file %s" % classes_file)
            f = open(Path(classes_file), 'w')
            for i in range(len(self.classes)):
                f.write(self.classes[i] + '\n')
            f.close()
        if fields_to_merge is not None:
            if merged_field is not None:
                print("Merging fields <<", fields_to_merge, ">> to new field <<", merged_field, ">>")
                self._merge_data(fields_to_merge=fields_to_merge.split(' '), merged_field=merged_field)
            else:
                raise IOError("Given fields to merge BUT not given name of merged field")

        if field_to_split is not None:
            if splitted_fields is not None:
                print("Splitting field <<", field_to_split, ">> to new fields <<", splitted_fields, ">>")
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
                                                                     test_size=len(data_to_div) -
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
