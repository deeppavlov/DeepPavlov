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
from typing import List

import numpy as np
import pickle
from deeppavlov.core.models.serializable import Serializable

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from pathlib import Path


@register('answer_generation_rus')
class AnswerGeneration(Component, Serializable):
    """
       Class for generation of answer using triplets with the entity
       in the question and relations predicted from the question by the
       relation prediction model.
       We search a triplet with the predicted relations
    """
    
    def __init__(self, load_path: str, *args, **kwargs) -> None:
        super().__init__(save_path = None, load_path = load_path)
        self.load()

    def load(self) -> None:
        load_path = Path(self.load_path).expanduser()
        with open(load_path, 'rb') as fl:
            self.q_to_name = pickle.load(fl)

    def save(self):
        pass
    
    def __call__(self, classes: List[List[str]],
                 entity_triplets: List[List[List[str]]],
                 *args, **kwargs) -> List[str]:
        
        objects_batch = []
        for n, rel_list in enumerate(classes):
            found = False
            for predicted_relation in rel_list:
                for rel_triplets in entity_triplets[n]:
                    relation_from_wiki = rel_triplets[0]
                    if predicted_relation == relation_from_wiki:
                        obj = rel_triplets[1]
                        found = True
                if found == True:
                    break
            
            objects_batch.append(obj)

        word_batch = []
        
        for obj in objects_batch:
            if obj in self.q_to_name:
                word = self.q_to_name[obj]["name"]
                word_batch.append(word)

        return word_batch

