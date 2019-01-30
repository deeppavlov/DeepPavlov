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

import numpy as np
from typing import List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
import pickle
from pathlib import Path

from collections import defaultdict
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords


@register('entity_linking')
class EntityLinking(Component):
    """
        Class for linking the words in the question and the corresponding entity
        in Freebase, then extracting triplets from Freebase with the entity
    """
    
    def __init__(self, entities_load_path: str,
                 freebase_load_path: str,
                 *args, **kwargs) -> None:

        self.inverted_index = defaultdict(list)
        self.stopword = set(stopwords.words('english'))
        entities_load_path = Path(entities_load_path).expanduser()
        with open(entities_load_path, "rb") as handler:
            self.inverted_index = pickle.load(handler, encoding='latin1')
            self.inverted_index = defaultdict(str, self.inverted_index)

        self.entity_dict = defaultdict(list)
        freebase_load_path = Path(freebase_load_path).expanduser()
        with open(freebase_load_path) as fl:
            line = fl.readline()
            split = line.strip('\n').split('\t')
            self.entity_dict[split[0]].append([split[1], split[2]])
            count = 0
            total = 0
            while line:
                total += 1
                line = fl.readline()
                split = line.strip('\n').split('\t')
                if len(split) > 2:
                    self.entity_dict[split[0]].append([split[1], split[2]])
                if len(split) < 3:
                    count += 1


    def get_ngram(self, text: str) -> List[str]:
        ngram = []
        tokens = text.split()
        for i in range(len(tokens)+1):
            for j in range(i):
                if i-j <= 3:
                    temp = " ".join(tokens[j:i])
                    if temp not in ngram:
                        ngram.append(temp)
    
        ngram = sorted(ngram, key=lambda x: len(x.split()), reverse=True)
        return ngram
    
    def __call__(self, texts: List[List[str]],
                 tags: List[List[int]],
                  *args, **kwargs) -> List[List[List[str]]]:
        entities = []
        for i, text in enumerate(texts):
            entity = ""
            for j, tok in enumerate(text):
                if tags[i][j] != 0:
                    entity += tok
                    entity += " "
            entity = entity[:-1].lower()
            entities.append(entity)

        link_scores_entity = []

        for entity in entities:
            link_scores = self.find_entity(entity)
            link_scores_entity.append(link_scores[:30])

        entity_triplets = []
        for link_scores in link_scores_entity:
            triplets = []
            for link_score in link_scores:
                entity_to_search = "www.freebase.com/m/" + link_score[0][0].split('.')[-1]
                triplet_list = self.entity_dict[entity_to_search]
                for triplet in triplet_list:
                    triplets.append([triplet, link_score[1]])
            entity_triplets.append(triplets)

        return entity_triplets
    

    def find_entity(self, entity: str) -> List[Tuple]:
        C = []
        C_scored = []
        tokens = self.get_ngram(entity)

        if len(tokens) > 0:
            maxlen = len(tokens[0].split())
        for item in tokens:
            if len(item.split()) < maxlen and len(C) == 0:
                maxlen = len(item.split())
            if len(item.split()) < maxlen and len(C) > 0:
                break
            if item in self.stopword:
                continue
            C.extend(self.inverted_index[item])


        for mid_text_type in sorted(set(C)):
            score = fuzz.ratio(mid_text_type[1], entity) / 100.0
            C_scored.append((mid_text_type, score))

        C_scored.sort(key = lambda x: x[1], reverse = True)
    
        return C_scored
