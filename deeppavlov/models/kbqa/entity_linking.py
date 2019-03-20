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
import itertools

from fuzzywuzzy import fuzz
import pymorphy2
import nltk

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.models.component import Component
from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher

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

    LANGUAGES = set(['rus'])

    def __init__(self, load_path: str, wiki_filename: str, entities_filename: str, inverted_index_filename: str,
                 id_to_name_file: str, lemmatize: bool = True, debug: bool = False, rule_filter_entities: bool = True,
                 use_inverted_index: bool = True, language: str = 'rus', *args, **kwargs) -> None:
        """

        Args:
            load_path: path to folder with wikidata files
            wiki_filename: file with Wikidata triplets
            entities_filename: file with dict of entity titles (keys) and entity ids (values)
            inverted_index_filename: file with dict of words (keys) and entities containing these words (values)
            id_to_name_file: file with dict of entity ids (keys) and entities names and aliases (values)
            lemmatize: whether to lemmatize tokens of extracted entity
            debug: whether to print entities extracted from Wikidata
            rule_filter_entities: whether to filter entities which do not fit the question
            use_inverted_index: whether to use inverted index for entity linking
            language - the language of the linker (used for filtration of some questions to improve overall performance)
            *args:
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.debug = debug
        self.rule_filter_entities = rule_filter_entities
        self.use_inverted_index = use_inverted_index
        self._language = language
        if language not in self.LANGUAGES:
            log.warning(f'EntityLinker supports only the following languages: {self.LANGUAGES}')

        self._wiki_filename = wiki_filename
        self._entities_filename = entities_filename
        self.inverted_index_filename = inverted_index_filename
        self.id_to_name_file = id_to_name_file

        self.name_to_q: Optional[Dict[str, List[Tuple[str]]]] = None
        self.wikidata: Optional[Dict[str, List[List[str]]]] = None
        self.inverted_index: Optional[Dict[str, List[Tuple[str]]]] = None
        self.id_to_name: Optional[Dict[str, Dict[List[str]]]] = None
        self.load()
        if self.use_inverted_index:
            alphabet = "abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя1234567890-_()=+!?.,/;:&@<>|#$%^*"
            dictionary_words = list(self.inverted_index.keys())
            self.searcher = LevenshteinSearcher(alphabet, dictionary_words)

    def load(self) -> None:
        if self.use_inverted_index:
            with open(self.load_path / self.inverted_index_filename, 'rb') as inv:
                self.inverted_index = pickle.load(inv)
                self.inverted_index: Dict[str, List[Tuple[str]]]
            with open(self.load_path / self.id_to_name_file, 'rb') as i2n:
                self.id_to_name = pickle.load(i2n)
                self.id_to_name: Dict[str, Dict[List[str]]]
        else:
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
            if self.use_inverted_index:
                candidate_entities = self.candidate_entities_inverted_index(entity)
                candidate_names = self.candidate_entities_names(candidate_entities)
                wiki_entities, confidences, srtd_cand_ent = self.sort_found_entities(candidate_entities,
                                                                                     candidate_names, entity)
            else:
                candidate_entities = self.find_candidate_entities(entity)

                srtd_cand_ent = sorted(candidate_entities, key=lambda x: x[2], reverse=True)
                if len(srtd_cand_ent) > 0:
                    wiki_entities = [ent[1] for ent in srtd_cand_ent]
                    confidences = [1.0 for i in range(len(srtd_cand_ent))]
                    srtd_cand_ent = [(ent[0], ent[1], conf, ent[2]) for ent, conf in zip(srtd_cand_ent, confidences)]
                else:
                    candidates = self.fuzzy_entity_search(entity)
                    candidates = list(set(candidates))
                    srtd_cand_ent = [(ent[0][0], ent[0][1], ent[1], ent[0][2]) for ent in candidates]
                    srtd_cand_ent = sorted(srtd_cand_ent, key=lambda x: (x[2], x[3]), reverse=True)

                    if len(srtd_cand_ent) > 0:
                        wiki_entities = [ent[1] for ent in srtd_cand_ent]
                        confidences = [float(ent[2]) * 0.01 for ent in srtd_cand_ent]
                    else:
                        wiki_entities = ["None"]
                        confidences = [0.0]

        entity_triplets = self.extract_triplets_from_wiki(wiki_entities)
        if self.rule_filter_entities and self._language == 'rus':
            filtered_entities, filtered_entity_triplets = self.filter_triplets_rus(entity_triplets,
                                                                                   question_tokens, srtd_cand_ent)
        if self.debug:
            self._log_entities(filtered_entities[:10])

        return filtered_entity_triplets, confidences

    def _log_entities(self, srtd_cand_ent):
        entities_to_print = []
        for name, q, ratio, n_rel in srtd_cand_ent:
            entities_to_print.append(f'{name}, http://wikidata.org/wiki/{q}, {ratio}, {n_rel}')
        log.debug('\n'+'\n'.join(entities_to_print))

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
        candidate_entities = list(set(candidate_entities))

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
    def filter_triplets_rus(entity_triplets: List[List[List[str]]], question_tokens: List[str],
                            srtd_cand_ent: List[Tuple[str]]) -> Tuple[List[Tuple[str]], List[List[List[str]]]]:

        question = ' '.join(question_tokens).lower()
        what_template = 'что '
        found_what_template = False
        found_what_template = question.find(what_template) > -1
        filtered_entity_triplets = []
        filtered_entities = []
        for wiki_entity, triplets_for_entity in zip(srtd_cand_ent, entity_triplets):
            entity_is_human = False
            entity_is_asteroid = False
            entity_is_named = False
            entity_title = wiki_entity[0]
            if entity_title[0].isupper():
                entity_is_named = True
            property_is_instance_of = 'P31'
            id_for_entity_human = 'Q5'
            id_for_entity_asteroid = 'Q3863'
            for triplet in triplets_for_entity:
                if triplet[0] == property_is_instance_of and triplet[1] == id_for_entity_human:
                    entity_is_human = True
                    break
                if triplet[0] == property_is_instance_of and triplet[1] == id_for_entity_asteroid:
                    entity_is_asteroid = True
                    break
            if found_what_template and (entity_is_human or entity_is_named or entity_is_asteroid or wiki_entity[2]<90):
                continue
            filtered_entity_triplets.append(triplets_for_entity)
            filtered_entities.append(wiki_entity)

        return filtered_entities, filtered_entity_triplets

    def candidate_entities_inverted_index(self, entity: str) -> List[Tuple[str]]:
        word_tokens = nltk.word_tokenize(entity)
        candidate_entities = []

        for tok in word_tokens:
            if len(tok) > 1:
                found = False
                if tok in self.inverted_index:
                    candidate_entities += self.inverted_index[tok]
                    found = True
                morph_parse_tok = self.morph.parse(tok)[0]
                lemmatized_tok = morph_parse_tok.normal_form
                if lemmatized_tok != tok and lemmatized_tok in self.inverted_index:
                    candidate_entities += self.inverted_index[lemmatized_tok]
                    found = True
                if not found:
                    words_with_levens_1 = self.searcher.search(tok, d=1)
                    for word in words_with_levens_1:
                        candidate_entities += self.inverted_index[word[0]]
        candidate_entities = list(set(candidate_entities))

        return candidate_entities

    def candidate_entities_names(self, candidate_entities: List[Tuple[str]]) -> List[List[str]]:
        candidate_names = []
        for candidate in candidate_entities:
            entity_id = candidate[0]
            entity_names = [self.id_to_name[entity_id]["name"]]
            if "aliases" in self.id_to_name[entity_id].keys():
                aliases = self.id_to_name[entity_id]["aliases"]
                for alias in aliases:
                    entity_names.append(alias)
            candidate_names.append(entity_names)

        return candidate_names

    def sort_found_entities(self, candidate_entities: List[Tuple[str]],
                                  candidate_names: List[List[str]],
                                  entity: str) -> Tuple[List[str], List[str], List[Tuple[str]]]:
        entities_ratios = []
        for candidate, entity_names in zip(candidate_entities, candidate_names):
            entity_id = candidate[0]
            num_rels = candidate[1]
            entity_name = entity_names[0]
            morph_parse_entity = self.morph.parse(entity)[0]
            lemm_entity = morph_parse_entity.normal_form
            fuzz_ratio_lemm = max([fuzz.ratio(name.lower(), lemm_entity.lower()) for name in entity_names])
            fuzz_ratio_nolemm = max([fuzz.ratio(name.lower(), entity.lower()) for name in entity_names])
            fuzz_ratio = max(fuzz_ratio_lemm, fuzz_ratio_nolemm)
            entities_ratios.append((entity_name, entity_id, fuzz_ratio, num_rels))

        srtd_with_ratios = sorted(entities_ratios, key=lambda x: (x[2], x[3]), reverse=True)
        wiki_entities = [ent[1] for ent in srtd_with_ratios if ent[2] > 84]
        confidences = [float(ent[2])*0.01 for ent in srtd_with_ratios if ent[2] > 84]
        
        return wiki_entities, confidences, srtd_with_ratios
