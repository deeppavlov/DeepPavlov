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
from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
import pickle
from pathlib import Path

from fuzzywuzzy import fuzz
import pymorphy2

log = getLogger(__name__)


@register('entity_linking_wikidata')
class EntityLinkingWikidata(Component):
    """
        Class for linking the words in the question and the corresponding entity
        in Freebase, then extracting triplets from Freebase with the entity
    """

    def __init__(self, entities_load_path: str,
                 wiki_load_path: str,
                 debug: bool = True,
                 *args, **kwargs) -> None:

        entities_load_path = Path(entities_load_path).expanduser()
        with open(entities_load_path, "rb") as e:
            self.name_to_q = pickle.load(e)

        wiki_load_path = Path(wiki_load_path).expanduser()
        with open(wiki_load_path, "rb") as f:
            self.wikidata = pickle.load(f)

        self.morph = pymorphy2.MorphAnalyzer()
        self._debug = debug

    def __call__(self, texts: List[List[str]],
                 tags: List[List[int]],
                 *args, **kwargs) -> List[List[List[str]]]:

        text_entities = []
        for i, text in enumerate(texts):
            entity = []
            for j, tok in enumerate(text):
                if tags[i][j] != 0:
                    entity.append(tok)
            entity = ' '.join(entity)
            text_entities.append(entity)

        if self._debug:
            questions = [' '.join(t) for t in texts]
            log.debug(f'Questions: {questions}')
            log.debug(f'Entities extracted by NER: {text_entities}')

        wiki_entities_batch = []
        confidences = []

        for entity in text_entities:
            if not entity:
                wiki_entities_batch.append(["None"])
            else:
                candidate_entities = []
                candidate_entities += self.name_to_q.get(entity, [])
                entity_split = entity.split(' ')
                for tok in entity_split:
                    entity_lemm = []
                    for tok_2 in entity_split:
                        if tok_2 == tok:
                            morph_parse_tok = self.morph.parse(tok_2)[0]
                            lemmatized_tok = morph_parse_tok.normal_form
                            entity_lemm.append(lemmatized_tok)
                        else:
                            entity_lemm.append(tok_2)
                    entity_lemm = ' '.join(entity_lemm)
                    if entity_lemm != entity:
                        candidate_entities += self.name_to_q.get(entity_lemm, [])
                            
                srtd_cand_ent = sorted(candidate_entities, key=lambda x: x[2], reverse=True)
                if len(srtd_cand_ent) > 0:
                    wiki_entities_batch.append([srtd_cand_ent[i][1] for i in range(len(srtd_cand_ent))])
                    confidences.append([1.0 for i in range(len(srtd_cand_ent))])
                else:
                    word_length = len(entity)
                    candidates = []
                    for title in self.name_to_q:
                        length_ratio = len(title) / word_length
                        if length_ratio > 0.5 and length_ratio < 1.5:
                            ratio = fuzz.ratio(title, entity)
                            if ratio > 50:
                                entity_candidates = self.name_to_q.get(title, [])
                                for cand in entity_candidates:
                                    candidates.append((cand, fuzz.ratio(entity, cand[0])))
                    
                    candidates = list(set(candidates))
                    srtd_cand_ent = sorted(candidates, key=lambda x: x[1], reverse=True)
                    
                    if len(srtd_cand_ent) > 0:
                        wiki_entities_batch.append([srtd_cand_ent[i][0][1] for i in range(len(srtd_cand_ent))])
                        confidences.append([srtd_cand_ent[i][1]*0.01 for i in range(len(srtd_cand_ent))])
                    else:
                        wiki_entities_batch.append(["None"])
                        confidences.append([0.0])

        entity_triplets_batch = []
        for entity_ids in wiki_entities_batch:
            entity_triplets = []
            for entity_id in entity_ids:
                if entity_id in self.wikidata and entity_id.startswith('Q'):
                    entity_triplets.append(self.wikidata[entity_id])
                else:
                    entity_triplets.append([])
            entity_triplets_batch.append(entity_triplets)

        return entity_triplets_batch, confidences

