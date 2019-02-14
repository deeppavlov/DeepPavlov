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
import pymorphy2


@register('entity_linking')
class EntityLinking(Component):
    """
        Class for linking the words in the question and the corresponding entity
        in Freebase, then extracting triplets from Freebase with the entity
    """
    
    def __init__(self, entities_load_path: str,
                 wiki_load_path: str,
                 *args, **kwargs) -> None:

        entities_load_path = Path(entities_load_path).expanduser()
        with open(entities_load_path, "rb") as e:
            self.name_to_q = pickle.load(e)

        wiki_load_path = Path(wiki_load_path).expanduser()
        with open(wiki_load_path, "rb") as f:
            self.wikidata = pickle.load(f)

        self.morph = pymorphy2.MorphAnalyzer()
    
    def __call__(self, texts: List[List[str]],
                 tags: List[List[int]],
                  *args, **kwargs) -> List[List[List[str]]]:

        text_entities = []
        for i, text in enumerate(texts):
            entity = ""
            for j, tok in enumerate(text):
                if tags[i][j] != 0:
                    entity += tok
                    entity += " "
            entity = entity[:-1]
            text_entities.append(entity)


        wiki_entities = []
        confidences = []
     
        for entity in text_entities:
            if not entity:
                wiki_entities.append("None")
            else:
                morph_parse_entity = self.morph.parse(entity)[0]
                lemmatized_entity = morph_parse_entity.normal_form
                if entity[0].isupper():
                    lemmatized_entity = lemmatized_entity.capitalize()
                candidate_entities = self.name_to_q[lemmatized_entity]
                srtd_cand_ent = sorted(candidate_entities, key = lambda x: x[2], reverse = True)
                if len(srtd_cand_ent) > 0:
                    wiki_entities.append(srtd_cand_ent[0][1])
                if len(srtd_cand_ent) == 0:
                    candidates = []
                    for title in self.name_to_q:
                        ratio = fuzz.ratio(title, lemmatized_entity)
                        if ratio > 75:
                            candidates += self.name_to_q[title]
                    candidates = list(set(candidates))
                    srtd_cand_ent = sorted(candidates, key = lambda x: x[2], reverse = True)
                    if len(srtd_cand_ent) > 0:
                        wiki_entities.append(srtd_cand_ent[0][1])
                    else:
                        wiki_entities.append("None")

        entity_triplets = []
        for entity_id in wiki_entities:
            if entity_id in self.wikidata:
                entity_triplets.append(self.wikidata[entity_id])
            else:
                entity_triplets.append([])


        return entity_triplets
    
