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


import json
from random import Random
from typing import Dict, Any, List, Tuple, Generator, Optional
from collections import defaultdict
from logging import getLogger

import numpy as np
from overrides import overrides
from pyparsing import null_debug_action

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

import wandb


log = getLogger(__name__)

@register('few_shot_iterator')
class FewShotIterator(DataLearningIterator):

    def __init__(self,
                 data: Dict[str, List[Tuple[Any, Any]]],
                 seed: int = None, 
                 shuffle: bool = True, 
                 shot: Optional[int] = None,
                 save_path: Optional[str] = None,
                 *args, **kwargs) -> None:
        self.shot = shot
        self.shuffle = shuffle
        self.random = Random(seed)

        train_shot_examples = self._get_shot_examples(data.get('train', []))
        self.train = self.preprocess(train_shot_examples, *args, **kwargs)
        self.valid = self.preprocess(data.get('valid', []), *args, **kwargs)
        self.test = self.preprocess(data.get('test', []), *args, **kwargs)
        self.data = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test,
            'all': self.train + self.test + self.valid
        }


        if save_path is None:
            return
        
        with open(save_path, "w") as file:
            json_dict = {"columns": ["text","category"]}
            json_dict["data"] = [[text, label] for text, label in train_shot_examples]
            json.dump(json_dict, file, indent=4)
    
    def _gather_info(self, data: List[Tuple[Any, Any]]):
        unique_labels = list(set([label for text, label in data]))

        label2examples = {}
        for label in unique_labels:
            label2examples[label] = []
        for text, label in data:
            label2examples[label].append(text)
        
        label2negative = {}
        for i, label in enumerate(unique_labels):
            label2negative[label] = unique_labels.copy()
            del label2negative[label][i]
        
        return label2examples, label2negative


    @overrides
    def preprocess(self, data: List[Tuple[Any, Any]], *args, **kwargs) -> List[Tuple[Any, Any]]:
        if len(data) == 0:
            return data

        label2examples, label2negative = self._gather_info(data)
        
        nli_triplets = []
        # negative examples
        for text, label in data:
            for negative_label in label2negative[label]:
                for negative_example in label2examples[negative_label]:
                    nli_triplets.append([[text, negative_example], 0])
                    
        # positive examples
        for text, label in data:
            for positive_example in label2examples[label]:
                if positive_example != text:
                    nli_triplets.append([[text, positive_example], 1])

        if self.shuffle:
            self.random.shuffle(nli_triplets)
        return nli_triplets
        

    def _get_shot_examples(self, data: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        if self.shot is None:
            return data
        
        # shuffle data to select shot-examples
        if self.shuffle:
            self.random.shuffle(data)

        data_dict = {}
        for _, label in data:
            data_dict[label] = []

        for text, label in data:
            if len(data_dict[label]) < self.shot:
                data_dict[label].append(text)
        
        if max(len(x) for x in data_dict.values()) < self.shot:
            log.warning(f"Some labels have less than \"shot\"={self.shot} examples")

        new_data = []
        for label in data_dict.keys():
            for text in data_dict[label]:
                new_data.append((text, label))
        return new_data
