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
import pickle
from logging import getLogger
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import itertools

from fuzzywuzzy import fuzz
import pymorphy2

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('entity_linker')
class EntityLinker(Component, Serializable):
    """
        This class extracts from Wikidata candidate entities for the entity mentioned in the question and then extracts
        triplets from Wikidata for the extracted entity. Candidate entities are searched in the dictionary where keys
        are titles and aliases of Wikidata entities and values are lists of tuples (entity_title, entity_id,
        number_of_relations). First candidate entities are searched in the dictionary by keys where the keys are
        entities extracted from the question, if nothing is found entities are searched in the dictionary using
        Levenstein distance between the entity and keys (titles) in the dictionary.
    """
    def __init__(self, load_path: str, wiki_filename: str, entities_filename: str, lemmatize: bool = True,
                 debug: bool = False, rule_filter_entities: bool = True, substring_entity_search: bool = False,
                 *args, **kwargs) -> None:
        """

        Args:
            load_path: path to folder with wikidata files
            wiki_filename: file with Wikidata triplets
            entities_filename: file with dict of entity titles (keys) and entity ids (values)
            lemmatize: whether to lemmatize tokens of extracted entity
            debug: whether to print entities extracted from Wikidata
            rule_filter_entities: whether to filter entities which do not fit the question
            substring_entity_search: whether to look for entities in Wikidata which contain entity extracted from the
            question as a substring
            *args:
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.debug = debug
        self.rule_filter_entities = rule_filter_entities
        self.substring_entity_search = substring_entity_search

        self._wiki_filename = wiki_filename
        self._entities_filename = entities_filename

        self.name_to_q: Optional[Dict[str, List[Tuple[str]]]] = None
        self.wikidata: Optional[Dict[str, List[List[str]]]] = None
        self.load()

    def load(self) -> None:
        with open(self.load_path / self._entities_filename, 'rb') as e:
            self.name_to_q = pickle.load(e)
            self.name_to_q: Dict[str, List[Tuple[str]]]
        with open(self.load_path / self._wiki_filename, 'rb') as w:
            self.wikidata = pickle.load(w)
            self.wikidata: Dict[str, List[List[str]]]

    def save(self) -> None:
        pass

    def __call__(self, entity: str, question_tokens: List[str]) -> Tuple[List[List[List[str]]], List[str]]:
        confidences = []
        srtd_cand_ent = []
        if not entity:
            wiki_entities = ['None']
        else:
            candidate_entities = self.find_candidate_entities(entity)

            srtd_cand_ent = sorted(candidate_entities, key=lambda x: x[2], reverse=True)
            if len(srtd_cand_ent) > 0:
                wiki_entities = [srtd_cand_ent[i][1] for i in range(len(srtd_cand_ent))]

                confidences = [1.0 for i in range(len(srtd_cand_ent))]
            else:
                if self.substring_entity_search:
                    candidates = self.substring_entity_search(entity)
                    candidates = list(set(candidates))
                    srtd_cand_ent = sorted(candidates, key=lambda x: x[2], reverse=True)
                    if len(srtd_cand_ent) > 0:
                        wiki_entities = [srtd_cand_ent[i][1] for i in range(len(srtd_cand_ent))]
                else:
                    candidates = self.fuzzy_entity_search(entity)
                    candidates = list(set(candidates))
                    srtd_cand_ent_with_scores = sorted(candidates, key=lambda x: x[1], reverse=True)
                    srtd_cand_ent = [srtd_cand_ent_with_scores[i][0] for i in range(len(srtd_cand_ent_with_scores))]

                    if len(srtd_cand_ent_with_scores) > 0:
                        wiki_entities = [srtd_cand_ent_with_scores[i][0][1] for i in range(len(srtd_cand_ent))]
                        confidences = [float(srtd_cand_ent_with_scores[i][1]) * 0.01 for i in range(len(srtd_cand_ent))]
                    else:
                        wiki_entities = ["None"]
                        confidences = [0.0]
        if self.debug:
            self._log_entities(srtd_cand_ent[:10])
        entity_triplets = self.extract_triplets_from_wiki(wiki_entities)
        if self.rule_filter_entities:
            filtered_entity_triplets = self.filter_triplets(entity_triplets, question_tokens, srtd_cand_ent)

        return filtered_entity_triplets, confidences

    def _log_entities(self, srtd_cand_ent):
        entities_to_print = []
        for name, q, n_rel in srtd_cand_ent:
            entities_to_print.append(f'{name}, http://wikidata.org/wiki/{q[1]}, {n_rel[2]}')
        log.debug('\n'.join(entities_to_print))

    def find_candidate_entities(self, entity: str) -> List[str]:
        candidate_entities = list(self.name_to_q.get(entity, []))
        entity_split = entity.split(' ')
        if len(entity_split) < 6 and self.lemmatize:
            entity_lemm_tokens = []
            for tok in entity_split:
                morph_parse_tok = self.morph.parse(tok)[0]
                lemmatized_tok = morph_parse_tok.normal_form
                entity_lemm_tokens.append(lemmatized_tok)
            masks = itertools.product([False, True], repeat=len(entity_split))
            for mask in masks:
                entity_lemm = []
                for i in range(len(entity_split)):
                    if mask[i]:
                        entity_lemm.append(entity_split[i])
                    else:
                        entity_lemm.append(entity_lemm_tokens[i])
                entity_lemm = ' '.join(entity_lemm)
                if entity_lemm != entity:
                    candidate_entities += self.name_to_q.get(entity_lemm, [])

        return candidate_entities

    def fuzzy_entity_search(self, entity: str) -> List[Tuple[Tuple, str]]:
        word_length = len(entity)
        candidates = []
        for title in self.name_to_q:
            length_ratio = len(title) / word_length
            if length_ratio > 0.75 and length_ratio < 1.25:
                ratio = fuzz.ratio(title, entity)
                if ratio > 70:
                    entity_candidates = self.name_to_q.get(title, [])
                    for cand in entity_candidates:
                        candidates.append((cand, fuzz.ratio(entity, cand[0])))
        return candidates

    def substring_entity_search(self, entity: str) -> List[Tuple[str]]:
        entity_lower = entity.lower()
        candidates = []
        for title in self.name_to_q:
            if title.find(entity_lower) > -1:
                entity_candidates = self.name_to_q.get(title, [])
                for cand in entity_candidates:
                    candidates.append(cand)
        return candidates

    def extract_triplets_from_wiki(self, entity_ids: List[str]) -> List[List[List[str]]]:
        entity_triplets = []
        for entity_id in entity_ids:
            if entity_id in self.wikidata and entity_id.startswith('Q'):
                triplets_for_entity = self.wikidata[entity_id]
                entity_triplets.append(triplets_for_entity)
            else:
                entity_triplets.append([])

        return entity_triplets

    @staticmethod
    def filter_triplets(entity_triplets: List[List[List[str]]], question_tokens: List[str],
        srtd_cand_ent: List[Tuple[str]]) -> List[List[List[str]]]:

        question_begin = question_tokens[0].lower()
        what_template = 'что'
        filtered_entity_triplets = []
        for wiki_entity, triplets_for_entity in zip(srtd_cand_ent, entity_triplets):
            entity_is_human = False
            entity_is_named = False
            entity_title = wiki_entity[0]
            if entity_title[0].isupper():
                entity_is_named = True
            property_is_instance_of = 'P31'
            id_for_entity_human = 'Q5'
            for triplet in triplets_for_entity:
                if triplet[0] == property_is_instance_of and triplet[1] == id_for_entity_human:
                    entity_is_human = True
                    break
            if question_begin == what_template and (entity_is_human or entity_is_named):
                continue
            filtered_entity_triplets.append(triplets_for_entity)

        return filtered_entity_triplets
