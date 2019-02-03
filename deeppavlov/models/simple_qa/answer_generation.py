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
from deeppavlov.core.models.serializable import Serializable

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from pathlib import Path


@register('answer_generation')
class AnswerGeneration(Component, Serializable):
    """
       Class for generation of answer using triplets with the entity
       in the question and relations predicted from the question by the
       relation prediction model.
       We search a triplet with the predicted relations
    """
    
    def __init__(self, load_path: str, *args, **kwargs) -> None:
        super().__init__(save_path=None, load_path=load_path)
        self.names_dict = None
        self.load()

    def load(self) -> None:
        load_path = Path(self.load_path).expanduser()
        with open(load_path, 'r') as fl:
            lines = fl.readlines()
            self.names_dict = {}
            for line in lines:
                fb_id = line.strip('\n').split('\t')[0]
                name = line.strip('\n').split('\t')[1]
                self.names_dict[fb_id] = name

    def save(self):
        pass
    
    def __call__(self, classes: List[List[str]],
                 entity_triplets: List[List[List[str]]],
                 *args, **kwargs) -> List[str]:
        
        objects_batch = []
        for n, rel_list in enumerate(classes):
            found = False
            for relation in rel_list:
                for triplet in entity_triplets[n]:
                    if triplet[0][0].split('com/')[1] == relation.split(':')[1].replace('.', '/'):
                        found_object = triplet[0][1].split(' ')[0]
                        obj = found_object.split('/')[-1]
                        found = True
                        break
                if found:
                    break
            if not found:
                for relation in rel_list:
                    for triplet in entity_triplets[n]:
                        base_rel = triplet[0][0].split('com/')[1]
                        found_rel = relation.split(':')[1]
                        if base_rel.split('/')[-1] == found_rel.split('.')[-1]:
                            found_object = triplet[0][1].split(' ')[0]
                            obj = found_object.split('/')[-1]
                            found = True
                            break
                    if found:
                        break
            objects_batch.append(obj)

        word_batch = []

        # Convert id to words
        for obj in objects_batch:
            if ("fb:m."+obj) in self.names_dict:
                word = self.names_dict[("fb:m."+obj)]
            word_batch.append(word)

        return word_batch

