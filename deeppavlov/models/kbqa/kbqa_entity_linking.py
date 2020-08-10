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

import re
import sqlite3
from logging import getLogger
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import nltk
import pymorphy2
from nltk.corpus import stopwords
from rapidfuzz import fuzz
from hdt import HDTDocument

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import load_pickle, save_pickle
from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher
from deeppavlov.models.kbqa.rel_ranking_bert_infer import RelRankerBertInfer

log = getLogger(__name__)


@register('kbqa_entity_linker')
class KBEntityLinker(Component, Serializable):
    """
        This class extracts from the knowledge base candidate entities for the entity mentioned in the question and then
        extracts triplets from Wikidata for the extracted entity. Candidate entities are searched in the dictionary
        where keys are titles and aliases of Wikidata entities and values are lists of tuples (entity_title, entity_id,
        number_of_relations). First candidate entities are searched in the dictionary by keys where the keys are
        entities extracted from the question, if nothing is found entities are searched in the dictionary using
        Levenstein distance between the entity and keys (titles) in the dictionary.
    """

    def __init__(self, load_path: str,
                 inverted_index_filename: str,
                 entities_list_filename: str,
                 q2name_filename: str,
                 save_path: str = None,
                 q2descr_filename: str = None,
                 rel_ranker: RelRankerBertInfer = None,
                 build_inverted_index: bool = False,
                 kb_format: str = "hdt",
                 kb_filename: str = None,
                 label_rel: str = None,
                 descr_rel: str = None,
                 aliases_rels: List[str] = None,
                 sql_table_name: str = None,
                 sql_column_names: List[str] = None,
                 lang: str = "en",
                 use_descriptions: bool = False,
                 lemmatize: bool = False,
                 use_prefix_tree: bool = False,
                 **kwargs) -> None:
        """

        Args:
            load_path: path to folder with inverted index files
            save_path: path where to save inverted index files
            inverted_index_filename: file with dict of words (keys) and entities containing these words
            entities_list_filename: file with the list of entities from the knowledge base
            q2name_filename: name of file which maps entity id to name
            q2descr_filename: name of file which maps entity id to description
            rel_ranker: component deeppavlov.models.kbqa.rel_ranker_bert_infer
            build_inverted_index: if "true", inverted index of entities of the KB will be built
            kb_format: "hdt" or "sqlite3"
            kb_filename: file with the knowledge base, which will be used for building of inverted index
            label_rel: relation in the knowledge base which connects entity ids and entity titles
            descr_rel: relation in the knowledge base which connects entity ids and entity descriptions
            aliases_rels: list of relations which connect entity ids and entity aliases
            sql_table_name: name of the table with the KB if the KB is in sqlite3 format
            sql_column_names: names of columns with subject, relation and object
            lang: language used
            use_descriptions: whether to use context and descriptions of entities for entity ranking
            lemmatize: whether to lemmatize tokens of extracted entity
            use_prefix_tree: whether to use prefix tree for search of entities with typos in entity labels
            **kwargs:
        """
        super().__init__(save_path=save_path, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.use_prefix_tree = use_prefix_tree
        self.inverted_index_filename = inverted_index_filename
        self.entities_list_filename = entities_list_filename
        self.build_inverted_index = build_inverted_index
        self.q2name_filename = q2name_filename
        self.q2descr_filename = q2descr_filename
        self.kb_format = kb_format
        self.kb_filename = kb_filename
        self.label_rel = label_rel
        self.aliases_rels = aliases_rels
        self.descr_rel = descr_rel
        self.sql_table_name = sql_table_name
        self.sql_column_names = sql_column_names
        self.inverted_index: Optional[Dict[str, List[Tuple[str]]]] = None
        self.entities_index: Optional[List[str]] = None
        self.q2name: Optional[List[Tuple[str]]] = None
        self.lang_str = f"@{lang}"
        if self.lang_str == "@en":
            self.stopwords = set(stopwords.words("english"))
        elif self.lang_str == "@ru":
            self.stopwords = set(stopwords.words("russian"))
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.rel_ranker = rel_ranker
        self.use_descriptions = use_descriptions

        if self.use_prefix_tree:
            alphabet = "!#%\&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz½¿ÁÄ" + \
                       "ÅÆÇÉÎÓÖ×ÚßàáâãäåæçèéêëíîïðñòóôöøùúûüýāăąćČčĐėęěĞğĩīİıŁłńňŌōőřŚśşŠšťũūůŵźŻżŽžơưșȚțəʻ" + \
                       "ʿΠΡβγБМавдежикмностъяḤḥṇṬṭầếờợ–‘’Ⅲ−∗"
            dictionary_words = list(self.inverted_index.keys())
            self.searcher = LevenshteinSearcher(alphabet, dictionary_words)

        if self.build_inverted_index:
            if self.kb_format == "hdt":
                self.doc = HDTDocument(str(expand_path(self.kb_filename)))
            elif self.kb_format == "sqlite3":
                self.conn = sqlite3.connect(str(expand_path(self.kb_filename)))
                self.cursor = self.conn.cursor()
            else:
                raise ValueError(f'unsupported kb_format value {self.kb_format}')
            self.inverted_index_builder()
            self.save()
        else:
            self.load()

    def load(self) -> None:
        self.inverted_index = load_pickle(self.load_path / self.inverted_index_filename)
        self.entities_list = load_pickle(self.load_path / self.entities_list_filename)
        self.q2name = load_pickle(self.load_path / self.q2name_filename)

    def save(self) -> None:
        save_pickle(self.inverted_index, self.save_path / self.inverted_index_filename)
        save_pickle(self.entities_list, self.save_path / self.entities_list_filename)
        save_pickle(self.q2name, self.save_path / self.q2name_filename)
        if self.q2descr_filename is not None:
            save_pickle(self.q2descr, self.save_path / self.q2descr_filename)

    def __call__(self, entity_substr_batch: List[List[str]], entity_positions_batch: List[List[List[int]]] = None,
                 context_tokens: List[List[str]] = None) -> Tuple[List[List[List[str]]], List[List[List[float]]]]:
        entity_ids_batch = []
        confidences_batch = []
        if entity_positions_batch is None:
            entity_positions_batch = [[[0] for i in range(len(entities_list))] for entities_list in entity_substr_batch]
        for entity_substr_list, entity_positions_list in zip(entity_substr_batch, entity_positions_batch):
            entity_ids_list = []
            confidences_list = []
            for entity_substr, entity_pos in zip(entity_substr_list, entity_positions_list):
                context = ""
                if self.use_descriptions:
                    context = ' '.join(context_tokens[:entity_pos[0]]+["[ENT]"]+context_tokens[entity_pos[-1]+1:])
                entity_ids, confidences = self.link_entity(entity_substr, context)
                entity_ids_list.append(entity_ids)
                confidences_list.append(confidences)
        entity_ids_batch.append(entity_ids_list)
        confidences_batch.append(confidences_list)

        return entity_ids_batch, confidences_batch

    def link_entity(self, entity: str, context: str = None) -> Tuple[List[str], List[float]]:
        confidences = []
        if not entity:
            entities_ids = ['None']
        else:
            candidate_entities = self.candidate_entities_inverted_index(entity)
            candidate_entities, candidate_names = self.candidate_entities_names(entity, candidate_entities)
            entities_ids, confidences, srtd_cand_ent = self.sort_found_entities(candidate_entities,
                                                                                candidate_names, entity, context)

        return entities_ids, confidences

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

    def sort_found_entities(self, candidate_entities: List[Tuple[int, str, int]],
                            candidate_names: List[List[str]],
                            entity: str,
                            context: str = None) -> Tuple[List[str], List[float], List[Tuple[str, str, int, int]]]:
        entities_ratios = []
        for candidate, entity_names in zip(candidate_entities, candidate_names):
            entity_num, entity_id, num_rels = candidate
            fuzz_ratio = max([fuzz.ratio(name.lower(), entity) for name in entity_names])
            entities_ratios.append((entity_num, entity_id, fuzz_ratio, num_rels))

        srtd_with_ratios = sorted(entities_ratios, key=lambda x: (x[2], x[3]), reverse=True)
        if self.use_descriptions:
            num_to_id = {entity_num: entity_id for entity_num, entity_id, _, _ in srtd_with_ratios[:30]}
            entity_numbers = [entity_num for entity_num, _, _, _ in srtd_with_ratios[:30]]
            scores = self.rel_ranker.rank_rels(context, entity_numbers)
            top_rels = [score[0] for score in scores]
            entity_ids = [num_to_id[num] for num in top_rels]
            confidences = [score[1] for score in scores]
        else:
            entity_ids = [ent[1] for ent in srtd_with_ratios]
            confidences = [float(ent[2]) * 0.01 for ent in srtd_with_ratios]

        return entity_ids, confidences, srtd_with_ratios

    def candidate_entities_names(self, entity: str,
                                 candidate_entities: List[Tuple[int, str, int]]) -> Tuple[List[Tuple[int, str, int]],
                                                                                          List[List[str]]]:
        entity_length = len(entity)
        candidate_names = []
        candidate_entities_filter = []
        for candidate in candidate_entities:
            entity_num = candidate[0]
            entity_names = []
            
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

    def inverted_index_builder(self) -> None:
        log.debug("building inverted index")
        entities_set = set()
        id_to_label_dict = defaultdict(list)
        id_to_descr_dict = {}
        label_to_id_dict = {}
        label_triplets = []
        alias_triplets_list = []
        descr_triplets = []
        if self.kb_format == "hdt":
            label_triplets, c = self.doc.search_triples("", self.label_rel, "")
            if self.aliases_rels is not None:
                for alias_rel in self.aliases_rels:
                    alias_triplets, c = self.doc.search_triples("", alias_rel, "")
                    alias_triplets_list.append(alias_triplets)
            if self.descr_rel is not None:
                descr_triplets, c = self.doc.search_triples("", self.descr_rel, "")

        if self.kb_format == "sqlite3":
            subject, relation, obj = self.sql_column_names
            query = f'SELECT {subject}, {relation}, {obj} FROM {self.sql_table_name} '\
                    f'WHERE {relation} = "{self.label_rel}";'
            res = self.cursor.execute(query)
            label_triplets = res.fetchall()
            if self.aliases_rels is not None:
                for alias_rel in self.aliases_rels:
                    query = f'SELECT {subject}, {relation}, {obj} FROM {self.sql_table_name} '\
                            f'WHERE {relation} = "{alias_rel}";'
                    res = self.cursor.execute(query)
                    alias_triplets = res.fetchall()
                    alias_triplets_list.append(alias_triplets)
            if self.descr_rel is not None:
                query = f'SELECT {subject}, {relation}, {obj} FROM {self.sql_table_name} '\
                        f'WHERE {relation} = "{self.descr_rel}";'
                res = self.cursor.execute(query)
                descr_triplets = res.fetchall()

        for triplets in [label_triplets] + alias_triplets_list:
            for triplet in triplets:
                entities_set.add(triplet[0])
                if triplet[2].endswith(self.lang_str):
                    label = triplet[2].replace(self.lang_str, '').replace('"', '')
                    id_to_label_dict[triplet[0]].append(label)
                    label_to_id_dict[label] = triplet[0]

        for triplet in descr_triplets:
            entities_set.add(triplet[0])
            if triplet[2].endswith(self.lang_str):
                descr = triplet[2].replace(self.lang_str, '').replace('"', '')
                id_to_descr_dict[triplet[0]].append(descr)

        popularities_dict = {}
        for entity in entities_set:
            if self.kb_format == "hdt":
                all_triplets, number_of_triplets = self.doc.search_triples(entity, "", "")
                popularities_dict[entity] = number_of_triplets
            if self.kb_format == "sqlite3":
                subject, relation, obj = self.sql_column_names
                query = f'SELECT COUNT({obj}) FROM {self.sql_table_name} WHERE {subject} = "{entity}";'
                res = self.cursor.execute(query)
                popularities_dict[entity] = res.fetchall()[0][0]

        entities_dict = {entity: n for n, entity in enumerate(entities_set)}
            
        inverted_index = defaultdict(list)
        for label in label_to_id_dict:
            tokens = re.findall(self.re_tokenizer, label.lower())
            for tok in tokens:
                if len(tok) > 1 and tok not in self.stopwords:
                    inverted_index[tok].append((entities_dict[label_to_id_dict[label]],
                                                popularities_dict[label_to_id_dict[label]]))
        self.inverted_index = dict(inverted_index)
        self.entities_list = list(entities_set)
        self.q2name = [id_to_label_dict[entity] for entity in self.entities_list]
        self.q2descr = []
        if id_to_descr_dict:
            self.q2descr = [id_to_descr_dict[entity] for entity in self.entities_list]
