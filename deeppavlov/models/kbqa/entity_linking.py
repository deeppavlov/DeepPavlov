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
from typing import List, Dict, Tuple, Optional, Any

import nltk
import pymorphy2
from fuzzywuzzy import fuzz

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher
from deeppavlov.models.kbqa.wiki_parser import WikiParser

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

    def __init__(self, load_path: str,
                 inverted_index_filename: str,
                 entities_list_filename: str,
                 q2name_filename: str,
                 wiki_parser: WikiParser = None,
                 use_hdt: bool = False,
                 lemmatize: bool = False,
                 use_prefix_tree: bool = False,
                 **kwargs) -> None:
        """

        Args:
            load_path: path to folder with wikidata files
            inverted_index_filename: file with dict of words (keys) and entities containing these words
            entities_list_filename: file with the list of entities from Wikidata
            q2name_filename: name of file which maps entity id to name
            wiki_parser: component deeppavlov.models.kbqa.wiki_parser
            use_hdt: whether to use hdt file with Wikidata
            lemmatize: whether to lemmatize tokens of extracted entity
            use_prefix_tree: whether to use prefix tree for search of entities with typos in entity labels
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.use_prefix_tree = use_prefix_tree
        self.use_hdt = use_hdt

        self.inverted_index_filename = inverted_index_filename
        self.entities_list_filename = entities_list_filename
        self.q2name_filename = q2name_filename
        self.inverted_index: Optional[Dict[str, List[Tuple[str]]]] = None
        if self.use_hdt:
            self.wiki_parser = wiki_parser
        self.load()

        if self.use_prefix_tree:
            alphabet = "!#%\&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz½¿ÁÄ" + \
                       "ÅÆÇÉÎÓÖ×ÚßàáâãäåæçèéêëíîïðñòóôöøùúûüýāăąćČčĐėęěĞğĩīİıŁłńňŌōőřŚśşŠšťũūůŵźŻżŽžơưșȚțəʻ" + \
                       "ʿΠΡβγБМавдежикмностъяḤḥṇṬṭầếờợ–‘’Ⅲ−∗"
            dictionary_words = list(self.inverted_index.keys())
            self.searcher = LevenshteinSearcher(alphabet, dictionary_words)

    def load(self) -> None:
        with open(self.load_path / self.inverted_index_filename, 'rb') as inv:
            self.inverted_index = pickle.load(inv)
            self.inverted_index: Dict[str, List[Tuple[str]]]
        with open(self.load_path / self.entities_list_filename, 'rb') as entlist:
            self.entities_list = pickle.load(entlist)
        if not self.use_hdt:
            with open(self.load_path / self.q2name_filename, 'rb') as q2name:
                self.q2name = pickle.load(q2name)

    def save(self) -> None:
        pass

    def __call__(self, entity: str) -> Tuple[List[str], List[float]]:
        confidences = []
        if not entity:
            wiki_entities = ['None']
        else:
            candidate_entities = self.candidate_entities_inverted_index(entity)
            candidate_entities, candidate_names = self.candidate_entities_names(entity, candidate_entities)
            wiki_entities, confidences, srtd_cand_ent = self.sort_found_entities(candidate_entities,
                                                                                 candidate_names, entity)

        return wiki_entities, confidences

    def candidate_entities_inverted_index(self, entity: str) -> List[Tuple[Any, Any, Any]]:
        word_tokens = nltk.word_tokenize(entity.lower())
        candidate_entities = []

        for tok in word_tokens:
            if len(tok) > 1:
                found = False
                if tok in self.inverted_index:
                    candidate_entities += self.inverted_index[tok]
                    found = True

                if self.lemmatize:
                    morph_parse_tok = self.morph.parse(tok)[0]
                    lemmatized_tok = morph_parse_tok.normal_form
                    if lemmatized_tok in self.inverted_index:
                        candidate_entities += self.inverted_index[lemmatized_tok]
                        found = True

                if not found and self.use_prefix_tree:
                    words_with_levens_1 = self.searcher.search(tok, d=1)
                    for word in words_with_levens_1:
                        candidate_entities += self.inverted_index[word[0]]
        candidate_entities = list(set(candidate_entities))
        candidate_entities = [(entity[0], self.entities_list[entity[0]], entity[1]) for entity in candidate_entities]

        return candidate_entities

    def sort_found_entities(self, candidate_entities: List[Tuple[str]],
                            candidate_names: List[List[str]],
                            entity: str) -> Tuple[List[str], List[float], List[Tuple[str, str, int, int]]]:
        entities_ratios = []
        for candidate, entity_names in zip(candidate_entities, candidate_names):
            entity_id = candidate[1]
            num_rels = candidate[2]
            entity_name = entity_names[0]
            fuzz_ratio = max([fuzz.ratio(name.lower(), entity.lower()) for name in entity_names])
            entities_ratios.append((entity_name, entity_id, fuzz_ratio, num_rels))

        srtd_with_ratios = sorted(entities_ratios, key=lambda x: (x[2], x[3]), reverse=True)
        wiki_entities = [ent[1] for ent in srtd_with_ratios]
        confidences = [float(ent[2]) * 0.01 for ent in srtd_with_ratios]

        return wiki_entities, confidences, srtd_with_ratios

    def candidate_entities_names(self, entity: str,
                                 candidate_entities: List[Tuple[int, str]]) -> Tuple[List[Tuple[str]], List[List[str]]]:
        entity_length = len(entity)
        candidate_names = []
        candidate_entities_filter = []
        for candidate in candidate_entities:
            entity_num = candidate[0]
            entity_id = candidate[1]
            entity_names = []
            if self.use_hdt:
                entity_name = self.wiki_parser("objects", "forw", entity_id, find_label=True)
                if entity_name != "Not Found" and len(entity_name) < 2 * entity_length:
                    entity_names.append(entity_name)
                    aliases = self.wiki_parser("objects", "forw", entity_id, find_alias=True)
                    for alias in aliases:
                        entity_names.append(alias)
                    candidate_names.append(entity_names)
                    candidate_entities_filter.append(candidate)
            else:
                entity_names_found = self.q2name[entity_num]
                if len(entity_names_found[0]) < 6 * entity_length:
                    entity_name = entity_names_found[0]
                    entity_names.append(entity_name)
                    if len(entity_names_found) > 1:
                        for alias in entity_names_found[1:]:
                            entity_names.append(alias)
                    candidate_names.append(entity_names)
                    candidate_entities_filter.append(candidate)

        return candidate_entities_filter, candidate_names
