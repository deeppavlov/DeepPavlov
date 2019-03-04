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

from fuzzywuzzy import fuzz
import pymorphy2
import itertools


class EntityLinker:
    def __init__(self, name_to_q, wikidata):
        self.name_to_q = name_to_q
        self.wikidata = wikidata
        self.morph = pymorphy2.MorphAnalyzer()

    def __call__(self, entity):

        if not entity:
            wiki_entities = ["None"]
        else:
            candidate_entities = self.find_candidate_entities(entity)

            srtd_cand_ent = sorted(candidate_entities, key=lambda x: x[2], reverse=True)
            if len(srtd_cand_ent) > 0:
                wiki_entities = [srtd_cand_ent[i][1] for i in range(len(srtd_cand_ent))]
                confidences = [1.0 for i in range(len(srtd_cand_ent))]
            else:
                candidates = self.substring_entity_search(entity, self.name_to_q)
                candidates = list(set(candidates))
                srtd_cand_ent = sorted(candidates, key=lambda x: x[2], reverse=True)
                if len(srtd_cand_ent) > 0:
                    candidates = self.fuzzy_entity_search(entity, self.name_to_q)
                    candidates = list(set(candidates))
                    srtd_cand_ent = sorted(candidates, key=lambda x: x[1], reverse=True)

                if len(srtd_cand_ent) > 0:
                    wiki_entities = [srtd_cand_ent[i][0][1] for i in range(len(srtd_cand_ent))]
                    confidences = [srtd_cand_ent[i][1]*0.01 for i in range(len(srtd_cand_ent))]
                else:
                    wiki_entities = ["None"]
                    confidences = [0.0]

        entity_triplets = self.extract_triplets_from_wiki(wiki_entities)
        return entity_triplets, confidences

    def find_candidate_entities(self, entity):
        candidate_entities = []
        candidate_entities += self.name_to_q.get(entity, [])
        entity_split = entity.split(' ')
        if len(entity_split) < 6:
            entity_lemm_tokens = []
            for tok in entity_split:
                morph_parse_tok = self.morph.parse(tok)[0]
                lemmatized_tok = morph_parse_tok.normal_form
                entity_lemm_tokens.append(lemmatized_tok)
            masks = itertools.product('01', repeat = len(entity_split))
            for mask in masks:
                entity_lemm = []
                for i in range(len(entity_split)):
                    if mask[i] == 0:
                        entity_lemm.append(entity_split[i])
                    else:
                        entity_lemm.append(entity_lemm_tokens[i])
                entity_lemm = ' '.join(entity_lemm)
                if entity_lemm != entity:
                    candidate_entities += self.name_to_q.get(entity_lemm, [])

        return candidate_entities

    def fuzzy_entity_search(self, entity):
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
        return candidates

    def substring_entity_search(self, entity):
        entity_lower = entity.lower()
        candidates = []
        for title in self.name_to_q:
            if title.find(entity_lower) > -1:
                entity_candidates = self.name_to_q.get(title, [])
                for cand in entity_candidates:
                    candidates.append(cand)
        return candidates

    def extract_triplets_from_wiki(self, entity_ids):
        entity_triplets = []
        for entity_id in entity_ids:
            if entity_id in self.wikidata and entity_id.startswith('Q'):
                entity_triplets.append(self.wikidata[entity_id])
            else:
                entity_triplets.append([])

        return entity_triplets
