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
from collections import defaultdict, Counter

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
                 types_dict_filename: Optional[str] = None,
                 who_entities_filename: Optional[str] = None,
                 save_path: str = None,
                 q2descr_filename: str = None,
                 descr_rank_score_thres: float = 0.01,
                 freq_dict_filename: Optional[str] = None,
                 entity_ranker: RelRankerBertInfer = None,
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
                 include_mention: bool = False,
                 num_entities_to_return: int = 5,
                 lemmatize: bool = False,
                 use_prefix_tree: bool = False,
                 **kwargs) -> None:
        """

        Args:
            load_path: path to folder with inverted index files
            inverted_index_filename: file with dict of words (keys) and entities containing these words
            entities_list_filename: file with the list of entities from the knowledge base
            q2name_filename: file which maps entity id to name
            types_dict_filename: file with types of entities
            who_entities_filename: file with the list of entities in Wikidata, which can be answers to questions
                with "Who" pronoun, i.e. humans, literary characters etc.
            save_path: path where to save inverted index files
            q2descr_filename: name of file which maps entity id to description
            descr_rank_score_thres: if the score of the entity description is less than threshold, the entity is not
                added to output list
            freq_dict_filename: filename with frequences dictionary of Russian words
            entity_ranker: component deeppavlov.models.kbqa.rel_ranker_bert_infer
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
            include_mention: whether to leave or delete entity mention from the sentence before passing to BERT ranker
            num_entities_to_return: how many entities for each substring the system returns
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
        self.types_dict_filename = types_dict_filename
        self.who_entities_filename = who_entities_filename
        self.q2descr_filename = q2descr_filename
        self.descr_rank_score_thres = descr_rank_score_thres
        self.freq_dict_filename = freq_dict_filename
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
        self.types_dict: Optional[Dict[str, List[str]]] = None
        self.lang_str = f"@{lang}"
        if self.lang_str == "@en":
            self.stopwords = set(stopwords.words("english"))
        elif self.lang_str == "@ru":
            self.stopwords = set(stopwords.words("russian"))
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.entity_ranker = entity_ranker
        self.use_descriptions = use_descriptions
        self.include_mention = include_mention
        self.num_entities_to_return = num_entities_to_return
        if self.use_descriptions and self.entity_ranker is None:
            raise ValueError("No entity ranker is provided!")

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

    def load_freq_dict(self, freq_dict_filename: str):
        with open(str(expand_path(freq_dict_filename)), 'r') as fl:
            lines = fl.readlines()
        pos_freq_dict = defaultdict(list)
        for line in lines:
            line_split = line.strip('\n').split('\t')
            if re.match("[\d]+\.[\d]+", line_split[2]):
                pos_freq_dict[line_split[1]].append((line_split[0], float(line_split[2])))
        nouns_with_freq = pos_freq_dict["s"]
        self.nouns_dict = {noun: freq for noun, freq in nouns_with_freq}

    def load(self) -> None:
        self.inverted_index = load_pickle(self.load_path / self.inverted_index_filename)
        self.entities_list = load_pickle(self.load_path / self.entities_list_filename)
        self.q2name = load_pickle(self.load_path / self.q2name_filename)
        if self.who_entities_filename:
            self.who_entities = load_pickle(self.load_path / self.who_entities_filename)
        if self.freq_dict_filename:
            self.load_freq_dict(self.freq_dict_filename)
        if self.types_dict_filename:
            self.types_dict = load_pickle(self.load_path / self.types_dict_filename)

    def save(self) -> None:
        save_pickle(self.inverted_index, self.save_path / self.inverted_index_filename)
        save_pickle(self.entities_list, self.save_path / self.entities_list_filename)
        save_pickle(self.q2name, self.save_path / self.q2name_filename)
        if self.q2descr_filename is not None:
            save_pickle(self.q2descr, self.save_path / self.q2descr_filename)

    def __call__(self, entity_substr_batch: List[List[str]],
                 templates_batch: List[str] = None,
                 context_batch: List[str] = None,
                 entity_types_batch: List[List[List[str]]] = None) -> Tuple[
        List[List[List[str]]], List[List[List[float]]]]:
        entity_ids_batch = []
        confidences_batch = []
        if templates_batch is None:
            templates_batch = ["" for _ in entity_substr_batch]
        if context_batch is None:
            context_batch = ["" for _ in entity_substr_batch]
        if entity_types_batch is None:
            entity_types_batch = [[[] for _ in entity_substr_list] for entity_substr_list in entity_substr_batch]
        for entity_substr_list, template_found, context, entity_types_list in \
                zip(entity_substr_batch, templates_batch, context_batch, entity_types_batch):
            entity_ids_list = []
            confidences_list = []
            for entity_substr, entity_types in zip(entity_substr_list, entity_types_list):
                entity_ids, confidences = self.link_entity(entity_substr, context, template_found, entity_types)
                if self.num_entities_to_return == 1:
                    if entity_ids:
                        entity_ids_list.append(entity_ids[0])
                        confidences_list.append(confidences[0])
                    else:
                        entity_ids_list.append("")
                        confidences_list.append(0.0)
                else:
                    entity_ids_list.append(entity_ids[:self.num_entities_to_return])
                    confidences_list.append(confidences[:self.num_entities_to_return])
            entity_ids_batch.append(entity_ids_list)
            confidences_batch.append(confidences_list)

        return entity_ids_batch, confidences_batch

    def link_entity(self, entity: str, context: Optional[str] = None, template_found: Optional[str] = None,
                    entity_types: List[str] = None, cut_entity: bool = False) -> Tuple[List[str], List[float]]:
        confidences = []
        if not entity:
            entities_ids = ['None']
        else:
            candidate_entities = self.candidate_entities_inverted_index(entity)
            if entity_types and self.types_dict:
                entity_types = set(entity_types)
                candidate_entities = [entity for entity in candidate_entities if
                                      self.types_dict.get(entity[1], set()).intersection(entity_types)]
            if cut_entity and candidate_entities and len(entity.split()) > 1 and candidate_entities[0][3] == 1:
                entity = self.cut_entity_substr(entity)
                candidate_entities = self.candidate_entities_inverted_index(entity)
            candidate_entities, candidate_names = self.candidate_entities_names(entity, candidate_entities)
            entities_ids, confidences, srtd_cand_ent = self.sort_found_entities(candidate_entities,
                                                                                candidate_names, entity, context)
            if template_found:
                entities_ids = self.filter_entities(entities_ids, template_found)

        return entities_ids, confidences

    def cut_entity_substr(self, entity: str):
        word_tokens = nltk.word_tokenize(entity.lower())
        word_tokens = [word for word in word_tokens if word not in self.stopwords]
        normal_form_tokens = [self.morph.parse(word)[0].normal_form for word in word_tokens]
        words_with_freq = [(word, self.nouns_dict.get(word, 0.0)) for word in normal_form_tokens]
        words_with_freq = sorted(words_with_freq, key=lambda x: x[1])
        return words_with_freq[0][0]

    def candidate_entities_inverted_index(self, entity: str) -> List[Tuple[Any, Any, Any]]:
        word_tokens = nltk.word_tokenize(entity.lower())
        word_tokens = [word for word in word_tokens if word not in self.stopwords]
        candidate_entities = []

        candidate_entities_for_tokens = []
        for tok in word_tokens:
            candidate_entities_for_tok = set()
            if len(tok) > 1:
                found = False
                if tok in self.inverted_index:
                    candidate_entities_for_tok = set(self.inverted_index[tok])
                    found = True

                if self.lemmatize:
                    if self.lang_str == "@ru":
                        morph_parse_tok = self.morph.parse(tok)[0]
                        lemmatized_tok = morph_parse_tok.normal_form
                    if self.lang_str == "@en":
                        lemmatized_tok = self.lemmatizer.lemmatize(tok)

                    if lemmatized_tok != tok and lemmatized_tok in self.inverted_index:
                        candidate_entities_for_tok = \
                            candidate_entities_for_tok.union(set(self.inverted_index[lemmatized_tok]))
                        found = True

                if not found and self.use_prefix_tree:
                    words_with_levens_1 = self.searcher.search(tok, d=1)
                    for word in words_with_levens_1:
                        candidate_entities_for_tok = \
                            candidate_entities_for_tok.union(set(self.inverted_index[word[0]]))
                candidate_entities_for_tokens.append(candidate_entities_for_tok)

        for candidate_entities_for_tok in candidate_entities_for_tokens:
            candidate_entities += list(candidate_entities_for_tok)
        candidate_entities = Counter(candidate_entities).most_common()
        candidate_entities = [(entity_num, self.entities_list[entity_num], entity_freq, count) for \
                              (entity_num, entity_freq), count in candidate_entities]

        return candidate_entities

    def sort_found_entities(self, candidate_entities: List[Tuple[int, str, int]],
                            candidate_names: List[List[str]],
                            entity: str,
                            context: str = None) -> Tuple[List[str], List[float], List[Tuple[str, str, int, int]]]:
        entities_ratios = []
        for candidate, entity_names in zip(candidate_entities, candidate_names):
            entity_num, entity_id, num_rels, tokens_matched = candidate
            fuzz_ratio = max([fuzz.ratio(name.lower(), entity) for name in entity_names])
            entities_ratios.append((entity_num, entity_id, tokens_matched, fuzz_ratio, num_rels))

        srtd_with_ratios = sorted(entities_ratios, key=lambda x: (x[2], x[3], x[4]), reverse=True)
        if self.use_descriptions:
            log.debug(f"context {context}")
            id_to_score = {entity_id: (tokens_matched, score) for _, entity_id, tokens_matched, score, _ in
                           srtd_with_ratios[:30]}
            entity_ids = [entity_id for _, entity_id, _, _, _ in srtd_with_ratios[:30]]
            scores = self.entity_ranker.rank_rels(context, entity_ids)
            entities_with_scores = [(entity_id, id_to_score[entity_id][0], id_to_score[entity_id][1], score) for
                                    entity_id, score in scores]
            entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[2], x[3]), reverse=True)
            entities_with_scores = [entity for entity in entities_with_scores if \
                                    (entity[3] > self.descr_rank_score_thres or entity[2] == 100.0)]
            log.debug(f"entities_with_scores {entities_with_scores[:10]}")
            entity_ids = [entity for entity, _, _, _ in entities_with_scores]
            confidences = [score for _, _, _, score in entities_with_scores]
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
            query = f'SELECT {subject}, {relation}, {obj} FROM {self.sql_table_name} ' \
                    f'WHERE {relation} = "{self.label_rel}";'
            res = self.cursor.execute(query)
            label_triplets = res.fetchall()
            if self.aliases_rels is not None:
                for alias_rel in self.aliases_rels:
                    query = f'SELECT {subject}, {relation}, {obj} FROM {self.sql_table_name} ' \
                            f'WHERE {relation} = "{alias_rel}";'
                    res = self.cursor.execute(query)
                    alias_triplets = res.fetchall()
                    alias_triplets_list.append(alias_triplets)
            if self.descr_rel is not None:
                query = f'SELECT {subject}, {relation}, {obj} FROM {self.sql_table_name} ' \
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

    def filter_entities(self, entities: List[str], template_found: str) -> List[str]:
        if template_found in ["who is xxx?", "who was xxx?"]:
            entities = [entity for entity in entities if entity in self.who_entities]
        if template_found in ["what is xxx?", "what was xxx?"]:
            entities = [entity for entity in entities if entity not in self.who_entities]
        return entities
