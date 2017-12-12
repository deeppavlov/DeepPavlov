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

import random
from typing import List, Dict, Generator, Tuple, Any
import numpy as np
from utils import preprocessing

# from deeppavlov.data.dataset import Dataset


class IntentDataset(object):
    def split(self, *args, **kwargs):
        pass

    def __init__(self, data, seed=None, *args, **kwargs):
        r""" Dataset takes a dict with fields 'train', 'test', 'valid'. A list of samples (pairs x, y)
        is stored in each field.

        Args:
            data: list of (x, y) pairs. Each pair is a sample from the dataset. x as well as y can be a tuple
                of different input features.
            seed (int): random seed for data shuffling. Defaults to None
        """
        rs = random.getstate()
        random.seed(seed)
        self.random_state = random.getstate()
        random.setstate(rs)

        self.train = data.get('train', [])
        self.valid = data.get('valid', [])
        self.test = data.get('test', [])
        self.split(*args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }

    def batch_generator(self, batch_size=64, data_type='train'):
        r"""This function returns a generator, which serves for generation of raw
        (no preprocessing such as tokenization) batches

        Args:
            batch_size (int): number of samples in batch
            data_type (str): can be either 'train', 'test', or 'valid'

        Returns:
            batch_gen (Generator): a generator,
            that iterates through the part (defined by data_type) of the dataset
        """
        data = self.data[data_type]
        data_len = len(data)
        order = list(range(data_len))

        rs = random.getstate()
        random.setstate(self.random_state)
        random.shuffle(order)
        self.random_state = random.getstate()
        random.setstate(rs)

        for i in range((data_len - 1) // batch_size + 1):
            yield list(zip(*[data[o] for o in order[i*batch_size:(i+1)*batch_size]]))

    def iter_all(self, data_type='train'):
        r"""Iterate through all data. It can be used for building dictionary or

        Args:
            data_type (str): can be either 'train', 'test', or 'valid'

        Returns:
            samples_gen: a generator, that iterates through the all samples in the selected data type of the dataset
        """
        # TODO: обработка ошибки, когда набора данных с таким названием не существует
        data = self.data[data_type]
        for x, y in data:
            yield (x, y)

    def extract_classes(self):
        intents = []
        all_data = self.iter_all(data_type='train')
        for sample in all_data:
            intents.extend(sample[1])
        intents = np.unique(intents)
        return np.array(intents)

    def preprocess(self, data_type='train', data=None):
        """Preprocess the data.

        Args:
            f: list of text samples

        Returns:
            preprocessed list of text samples
        """
        if data is not None:
            prep_data = preprocessing(data)
            return prep_data
        else:
            all_data = self.iter_all(data_type=data_type)
            texts = []
            for sample in all_data:
                texts.append(sample[0])
            prep_data = preprocessing(texts)
            return prep_data
