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

from logging import getLogger
from typing import List, Tuple

import numpy as np
import pickle
from deeppavlov.core.models.serializable import Serializable

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from pathlib import Path

log = getLogger(__name__)


@register('kb_answer_parser_wikidata')
class KBAnswerParserWikidata(Component, Serializable):
    """
       Class for generation of answer using triplets with the entity
       in the question and relations predicted from the question by the
       relation prediction model.
       We search a triplet with the predicted relations
    """
    
    def __init__(self, load_path: str, top_k_classes: int, classes_vocab_keys: Tuple,
                 debug: bool = False, *args, **kwargs) -> None:
        super().__init__(save_path=None, load_path=load_path)
        self.top_k_classes = top_k_classes
        self.classes = list(classes_vocab_keys)
        self._debug = debug
        self.load()

    def load(self) -> None:
        load_path = Path(self.load_path).expanduser()
        with open(load_path, 'rb') as fl:
            self.q_to_name = pickle.load(fl)

    def save(self):
        pass

    def __call__(self, relations_probs: List[List[str]],
                 entity_triplets_batch: List[List[List[str]]],
                 *args, **kwargs) -> List[str]:

        relations_batch = self._parse_relations_probs(relations_probs)
        if self._debug:
            log.debug(f'Top-k relations extracted: {relations_batch}')
        objects_batch = []
        for rel_list, entity_triplets in zip(relations_batch, entity_triplets_batch):
            found = False
            for predicted_relation in rel_list:
                for entities in entity_triplets:
                    for rel_triplets in entities:
                        relation_from_wiki = rel_triplets[0]
                        if predicted_relation == relation_from_wiki:
                            obj = rel_triplets[1]
                            found = True
                            break
                    if found == True:
                        break
                if found == True:
                    break
            if not found:
                obj = None
            objects_batch.append(obj)

        word_batch = []

        for obj in objects_batch:
            if obj in self.q_to_name:
                word = self.q_to_name[obj]["name"]
                word_batch.append(word)
            else:
                word_batch.append('Not Found')

        return word_batch

    def _parse_relations_probs(self, probas_batch: List[List[float]]) -> List[List[str]]:
        top_k_batch = []
        for probas in probas_batch:
            top_k_inds = np.asarray(probas).argsort()[-self.top_k_classes:][::-1]  # Make it top n and n to the __init__
            top_k_classes = [self.classes[k] for k in top_k_inds]
            top_k_batch.append(top_k_classes)
        return top_k_batch
