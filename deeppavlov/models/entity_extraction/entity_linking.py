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
from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict

import pymorphy2
from hdt import HDTDocument
from nltk.corpus import stopwords
from rapidfuzz import fuzz

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.commands.utils import expand_path

log = getLogger(__name__)


@register("entity_linker")
class EntityLinker(Component, Serializable):
    """
    Class for linking of entity substrings in the document to entities in Wikidata
    """

    def __init__(
            self,
            load_path: str,
            entities_database_filename: str,
            entity_ranker=None,
            num_entities_for_bert_ranking: int = 50,
            wikidata_file: str = None,
            num_entities_to_return: int = 10,
            max_text_len: int = 300,
            lang: str = "en",
            use_descriptions: bool = True,
            use_tags: bool = False,
            lemmatize: bool = False,
            full_paragraph: bool = False,
            use_connections: bool = False,
            max_paragraph_len: int = 250,
            **kwargs,
    ) -> None:
        """

        Args:
            load_path: path to folder with inverted index files
            entities_database_filename: file with sqlite database with Wikidata entities index
            entity_ranker: deeppavlov.models.torch_bert.torch_transformers_el_ranker.TorchTransformersEntityRankerInfer
            num_entities_for_bert_ranking: number of candidate entities for BERT ranking using description and context
            wikidata_file: .hdt file with Wikidata graph
            num_entities_to_return: number of candidate entities for the substring which are returned
            max_text_len: max length of context for entity ranking by description
            lang: russian or english
            use_description: whether to perform entity ranking by context and description
            use_tags: whether to use ner tags for entity filtering
            lemmatize: whether to lemmatize tokens
            full_paragraph: whether to use full paragraph for entity ranking by context and description
            use_connections: whether to ranking entities by number of connections in Wikidata
            max_paragraph_len: maximum length of paragraph for ranking by context and description
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.entities_database_filename = entities_database_filename
        self.num_entities_for_bert_ranking = num_entities_for_bert_ranking
        self.wikidata_file = wikidata_file
        self.entity_ranker = entity_ranker
        self.num_entities_to_return = num_entities_to_return
        self.max_text_len = max_text_len
        self.lang = f"@{lang}"
        if self.lang == "@en":
            self.stopwords = set(stopwords.words("english"))
        elif self.lang == "@ru":
            self.stopwords = set(stopwords.words("russian"))
        self.use_descriptions = use_descriptions
        self.use_connections = use_connections
        self.max_paragraph_len = max_paragraph_len
        self.use_tags = use_tags
        self.full_paragraph = full_paragraph
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.not_found_str = "not_in_wiki"

        self.load()

    def load(self) -> None:
        self.conn = sqlite3.connect(str(self.load_path / self.entities_database_filename))
        self.cur = self.conn.cursor()
        self.wikidata = None
        if self.wikidata_file:
            self.wikidata = HDTDocument(str(expand_path(self.wikidata_file)))

    def save(self) -> None:
        pass

    def __call__(
            self,
            entity_substr_batch: List[List[str]],
            entity_tags_batch: List[List[str]] = None,
            sentences_batch: List[List[str]] = None,
            entity_offsets_batch: List[List[List[int]]] = None,
            sentences_offsets_batch: List[List[Tuple[int, int]]] = None,
    ) -> Tuple[Union[List[List[List[str]]], List[List[str]]], Union[List[List[List[Any]]], List[List[Any]]],
               Union[List[List[List[str]]], List[List[str]]]]:
        if (not sentences_offsets_batch or sentences_offsets_batch[0] is None) and sentences_batch is not None \
                or not isinstance(sentences_offsets_batch[0][0], (list, tuple)):
            sentences_offsets_batch = []
            for sentences_list in sentences_batch:
                sentences_offsets_list = []
                start = 0
                for sentence in sentences_list:
                    end = start + len(sentence)
                    sentences_offsets_list.append([start, end])
                    start = end + 1
                sentences_offsets_batch.append(sentences_offsets_list)

        if entity_tags_batch is None or not entity_tags_batch[0]:
            entity_tags_batch = [["" for _ in entity_substr_list] for entity_substr_list in entity_substr_batch]
        else:
            entity_tags_batch = [[tag.upper() for tag in entity_tags] for entity_tags in entity_tags_batch]

        if sentences_batch is None:
            sentences_batch = [[] for _ in entity_substr_batch]
            sentences_offsets_batch = [[] for _ in entity_substr_batch]

        log.debug(f"sentences_batch {sentences_batch}")
        if (not entity_offsets_batch and sentences_batch) or not entity_offsets_batch[0] \
                or not isinstance(entity_offsets_batch[0][0], (list, tuple)):
            entity_offsets_batch = []
            for entity_substr_list, sentences_list in zip(entity_substr_batch, sentences_batch):
                text = " ".join(sentences_list).lower()
                log.debug(f"text {text}")
                entity_offsets_list = []
                for entity_substr in entity_substr_list:
                    st_offset = text.find(entity_substr.lower())
                    end_offset = st_offset + len(entity_substr)
                    entity_offsets_list.append([st_offset, end_offset])
                entity_offsets_batch.append(entity_offsets_list)

        entity_ids_batch, entity_conf_batch, entity_pages_batch = [], [], []
        for (entity_substr_list, entity_offsets_list, entity_tags_list, sentences_list, sentences_offsets_list,) in zip(
                entity_substr_batch,
                entity_offsets_batch,
                entity_tags_batch,
                sentences_batch,
                sentences_offsets_batch,
        ):
            entity_ids_list, entity_conf_list, entity_pages_list = self.link_entities(
                entity_substr_list,
                entity_offsets_list,
                entity_tags_list,
                sentences_list,
                sentences_offsets_list,
            )
            log.debug(f"entity_ids_list {entity_ids_list} entity_conf_list {entity_conf_list}")
            entity_ids_batch.append(entity_ids_list)
            entity_conf_batch.append(entity_conf_list)
            entity_pages_batch.append(entity_pages_list)
        return entity_ids_batch, entity_conf_batch, entity_pages_batch

    def link_entities(
            self,
            entity_substr_list: List[str],
            entity_offsets_list: List[List[int]],
            entity_tags_list: List[str],
            sentences_list: List[str],
            sentences_offsets_list: List[List[int]],
    ) -> Tuple[Union[List[List[str]], List[str]], Union[List[List[Any]], List[Any]], Union[List[List[str]], List[str]]]:
        log.debug(
            f"entity_substr_list {entity_substr_list} entity_tags_list {entity_tags_list} "
            f"entity_offsets_list {entity_offsets_list}"
        )
        entity_ids_list, conf_list, pages_list = [], [], []
        if entity_substr_list:
            entities_scores_list = []
            cand_ent_scores_list = []
            entity_substr_split_list = [
                [word for word in entity_substr.split(" ") if word not in self.stopwords and len(word) > 0]
                for entity_substr in entity_substr_list
            ]
            for entity_substr, entity_substr_split, tag in zip(
                    entity_substr_list, entity_substr_split_list, entity_tags_list
            ):
                cand_ent_scores = []
                if len(entity_substr) > 1:
                    entity_substr_split_lemm = [self.morph.parse(tok)[0].normal_form for tok in entity_substr_split]
                    cand_ent_init = self.find_exact_match(entity_substr, tag)
                    if not cand_ent_init or entity_substr_split != entity_substr_split_lemm:
                        cand_ent_init = self.find_fuzzy_match(entity_substr_split, tag)

                    for entity in cand_ent_init:
                        entities_scores = list(cand_ent_init[entity])
                        entities_scores = sorted(entities_scores, key=lambda x: (x[0], x[1]), reverse=True)
                        cand_ent_scores.append((entity, entities_scores[0]))
                    cand_ent_scores = sorted(cand_ent_scores, key=lambda x: (x[1][0], x[1][1]), reverse=True)

                cand_ent_scores = cand_ent_scores[:self.num_entities_for_bert_ranking]
                cand_ent_scores_list.append(cand_ent_scores)
                entity_ids = [elem[0] for elem in cand_ent_scores]
                entities_scores_list.append({ent: score for ent, score in cand_ent_scores})
                entity_ids_list.append(entity_ids)

            if self.use_connections:
                entity_ids_list = []
                entities_with_conn_scores_list = self.rank_by_connections(cand_ent_scores_list)
                for entities_with_conn_scores in entities_with_conn_scores_list:
                    entity_ids = [elem[0] for elem in entities_with_conn_scores]
                    entity_ids_list.append(entity_ids)

            entity_descr_list = []
            pages_dict = {}
            for entity_ids in entity_ids_list:
                entity_descrs = []
                for entity_id in entity_ids:
                    res = self.cur.execute("SELECT * FROM entity_labels WHERE entity='{}';".format(entity_id))
                    entity_info = res.fetchall()
                    if entity_info:
                        (
                            cur_entity_id,
                            cur_entity_label,
                            cur_entity_descr,
                            cur_entity_page,
                        ) = entity_info[0]
                        entity_descrs.append(cur_entity_descr)
                        pages_dict[cur_entity_id] = cur_entity_page
                    else:
                        entity_descrs.append("")
                entity_descr_list.append(entity_descrs)
            if self.use_descriptions:
                substr_lens = [len(entity_substr.split()) for entity_substr in entity_substr_list]
                entity_ids_list, conf_list = self.rank_by_description(
                    entity_substr_list,
                    entity_offsets_list,
                    entity_ids_list,
                    entity_descr_list,
                    entities_scores_list,
                    sentences_list,
                    sentences_offsets_list,
                    substr_lens,
                )
            if self.num_entities_to_return == 1:
                pages_list = [pages_dict.get(entity_ids, "") for entity_ids in entity_ids_list]
            else:
                pages_list = [[pages_dict.get(entity_id, "") for entity_id in entity_ids]
                              for entity_ids in entity_ids_list]

        return entity_ids_list, conf_list, pages_list

    def process_cand_ent(self, cand_ent_init, entities_and_ids, entity_substr_split, tag):
        if self.use_tags:
            for cand_entity_title, cand_entity_id, cand_entity_rels, cand_tag, *_ in entities_and_ids:
                if not tag or tag == cand_tag:
                    substr_score = self.calc_substr_score(cand_entity_title, entity_substr_split)
                    cand_ent_init[cand_entity_id].add((substr_score, cand_entity_rels))
            if not cand_ent_init:
                for cand_entity_title, cand_entity_id, cand_entity_rels, cand_tag, *_ in entities_and_ids:
                    substr_score = self.calc_substr_score(cand_entity_title, entity_substr_split)
                    cand_ent_init[cand_entity_id].add((substr_score, cand_entity_rels))
        else:
            for cand_entity_title, cand_entity_id, cand_entity_rels, *_ in entities_and_ids:
                substr_score = self.calc_substr_score(cand_entity_title, entity_substr_split)
                cand_ent_init[cand_entity_id].add((substr_score, cand_entity_rels))
        return cand_ent_init

    def find_title(self, entity_substr):
        entities_and_ids = []
        try:
            res = self.cur.execute("SELECT * FROM inverted_index WHERE title MATCH '{}';".format(entity_substr))
            entities_and_ids = res.fetchall()
        except sqlite3.OperationalError as e:
            log.debug(f"error in searching an entity {e}")
        return entities_and_ids

    def find_exact_match(self, entity_substr, tag):
        entity_substr_split = entity_substr.split()
        cand_ent_init = defaultdict(set)
        entities_and_ids = self.find_title(entity_substr)
        if entities_and_ids:
            cand_ent_init = self.process_cand_ent(cand_ent_init, entities_and_ids, entity_substr_split, tag)
        if entity_substr.startswith("the "):
            entity_substr = entity_substr.split("the ")[1]
            entity_substr_split = entity_substr_split[1:]
            entities_and_ids = self.find_title(entity_substr)
            cand_ent_init = self.process_cand_ent(cand_ent_init, entities_and_ids, entity_substr_split, tag)
        if self.lang == "@ru":
            entity_substr_split_lemm = [self.morph.parse(tok)[0].normal_form for tok in entity_substr_split]
            entity_substr_lemm = " ".join(entity_substr_split_lemm)
            if entity_substr_lemm != entity_substr:
                entities_and_ids = self.find_title(entity_substr_lemm)
                if entities_and_ids:
                    cand_ent_init = self.process_cand_ent(
                        cand_ent_init, entities_and_ids, entity_substr_split_lemm, tag
                    )
        return cand_ent_init

    def find_fuzzy_match(self, entity_substr_split, tag):
        if self.lang == "@ru":
            entity_substr_split_lemm = [self.morph.parse(tok)[0].normal_form for tok in entity_substr_split]
        else:
            entity_substr_split_lemm = entity_substr_split
        cand_ent_init = defaultdict(set)
        for word in entity_substr_split:
            part_entities_and_ids = self.find_title(word)
            cand_ent_init = self.process_cand_ent(cand_ent_init, part_entities_and_ids, entity_substr_split, tag)
            if self.lang == "@ru":
                word_lemm = self.morph.parse(word)[0].normal_form
                if word != word_lemm:
                    part_entities_and_ids = self.find_title(word_lemm)
                    cand_ent_init = self.process_cand_ent(
                        cand_ent_init,
                        part_entities_and_ids,
                        entity_substr_split_lemm,
                        tag
                    )
        return cand_ent_init

    def morph_parse(self, word):
        morph_parse_tok = self.morph.parse(word)[0]
        normal_form = morph_parse_tok.normal_form
        return normal_form

    def calc_substr_score(self, cand_entity_title, entity_substr_split):
        label_tokens = cand_entity_title.split()
        cnt = 0.0
        for ent_tok in entity_substr_split:
            found = False
            for label_tok in label_tokens:
                if label_tok == ent_tok:
                    found = True
                    break
            if found:
                cnt += 1.0
            else:
                for label_tok in label_tokens:
                    if label_tok[:2] == ent_tok[:2]:
                        fuzz_score = fuzz.ratio(label_tok, ent_tok)
                        if fuzz_score >= 80.0 and not found:
                            cnt += fuzz_score * 0.01
                            break
        substr_score = round(cnt / max(len(label_tokens), len(entity_substr_split)), 3)
        if len(label_tokens) == 2 and len(entity_substr_split) == 1:
            if entity_substr_split[0] == label_tokens[1]:
                substr_score = 0.5
            elif entity_substr_split[0] == label_tokens[0]:
                substr_score = 0.3
        return substr_score

    def rank_by_connections(self, cand_ent_scores_list: List[List[Union[str, Tuple[str, str]]]]):
        entities_for_ranking_list = []
        for entities_scores in cand_ent_scores_list:
            entities_for_ranking = []
            if entities_scores:
                max_score = entities_scores[0][1][0]
                for entity, scores in entities_scores:
                    if scores[0] == max_score:
                        entities_for_ranking.append(entity)
            entities_for_ranking_list.append(entities_for_ranking)

        entities_sets_list = []
        for entities_scores in cand_ent_scores_list:
            entities_sets_list.append({entity for entity, scores in entities_scores})

        entities_conn_scores_list = []
        for entities_scores in cand_ent_scores_list:
            cur_entity_dict = {}
            for entity, scores in entities_scores:
                cur_entity_dict[entity] = 0
            entities_conn_scores_list.append(cur_entity_dict)

        entities_objects_list, entities_triplets_list = [], []
        for entities_scores in cand_ent_scores_list:
            cur_objects_dict, cur_triplets_dict = {}, {}
            for entity, scores in entities_scores:
                objects, triplets = set(), set()
                tr, cnt = self.wikidata.search_triples(f"http://we/{entity}", "", "")
                for triplet in tr:
                    objects.add(triplet[2].split("/")[-1])
                    triplets.add((triplet[1].split("/")[-1], triplet[2].split("/")[-1]))
                cur_objects_dict[entity] = objects
                cur_triplets_dict[entity] = triplets
            entities_objects_list.append(cur_objects_dict)
            entities_triplets_list.append(cur_triplets_dict)

        already_ranked = {i: False for i in range(len(entities_for_ranking_list))}

        for i in range(len(entities_for_ranking_list)):
            for entity1 in entities_for_ranking_list[i]:
                for j in range(len(entities_for_ranking_list)):
                    if i != j and not already_ranked[j]:
                        inters = entities_objects_list[i][entity1].intersection(entities_sets_list[j])
                        if inters:
                            entities_conn_scores_list[i][entity1] += len(inters)
                            for entity2 in inters:
                                entities_conn_scores_list[j][entity2] += len(inters)
                            already_ranked[j] = True
                        else:
                            for entity2 in entities_triplets_list[j]:
                                inters = entities_triplets_list[i][entity1].intersection(
                                    entities_triplets_list[j][entity2]
                                )
                                inters = {elem for elem in inters if elem[0] != "P31"}
                                if inters:
                                    prev_score1 = entities_conn_scores_list[i].get(entity1, 0)
                                    prev_score2 = entities_conn_scores_list[j].get(entity2, 0)
                                    entities_conn_scores_list[i][entity1] = max(len(inters), prev_score1)
                                    entities_conn_scores_list[j][entity2] = max(len(inters), prev_score2)

        entities_with_conn_scores_list = []
        for i in range(len(entities_conn_scores_list)):
            entities_with_conn_scores_list.append(
                sorted(
                    list(entities_conn_scores_list[i].items()),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        return entities_with_conn_scores_list

    def rank_by_description(
            self,
            entity_substr_list: List[str],
            entity_offsets_list: List[List[int]],
            cand_ent_list: List[List[str]],
            cand_ent_descr_list: List[List[str]],
            entities_scores_list: List[Dict[str, Tuple[int, float]]],
            sentences_list: List[str],
            sentences_offsets_list: List[List[int]],
            substr_lens: List[int],
    ) -> Tuple[Union[List[List[str]], List[str]], Union[List[List[Any]], List[Any]]]:
        entity_ids_list = []
        conf_list = []
        contexts = []
        for (
                entity_substr,
                (entity_start_offset, entity_end_offset),
                candidate_entities,
        ) in zip(entity_substr_list, entity_offsets_list, cand_ent_list):
            sentence = ""
            rel_start_offset = 0
            rel_end_offset = 0
            found_sentence_num = 0
            for num, (sent, (sent_start_offset, sent_end_offset)) in enumerate(
                    zip(sentences_list, sentences_offsets_list)
            ):
                if entity_start_offset >= sent_start_offset and entity_end_offset <= sent_end_offset:
                    sentence = sent
                    found_sentence_num = num
                    rel_start_offset = entity_start_offset - sent_start_offset
                    rel_end_offset = entity_end_offset - sent_start_offset
                    break
            context = ""
            if sentence:
                start_of_sentence = 0
                end_of_sentence = len(sentence)
                if len(sentence) > self.max_text_len:
                    start_of_sentence = max(rel_start_offset - self.max_text_len // 2, 0)
                    end_of_sentence = min(rel_end_offset + self.max_text_len // 2, len(sentence))
                context = (
                        sentence[start_of_sentence:rel_start_offset] + "[ENT]" + sentence[
                                                                                 rel_end_offset:end_of_sentence]
                )
                if self.full_paragraph:
                    cur_sent_len = len(re.findall(self.re_tokenizer, context))
                    first_sentence_num = found_sentence_num
                    last_sentence_num = found_sentence_num
                    context = [context]
                    while True:
                        added = False
                        if last_sentence_num < len(sentences_list) - 1:
                            last_sentence_len = len(
                                re.findall(
                                    self.re_tokenizer,
                                    sentences_list[last_sentence_num + 1],
                                )
                            )
                            if cur_sent_len + last_sentence_len < self.max_paragraph_len:
                                context.append(sentences_list[last_sentence_num + 1])
                                cur_sent_len += last_sentence_len
                                last_sentence_num += 1
                                added = True
                        if first_sentence_num > 0:
                            first_sentence_len = len(
                                re.findall(
                                    self.re_tokenizer,
                                    sentences_list[first_sentence_num - 1],
                                )
                            )
                            if cur_sent_len + first_sentence_len < self.max_paragraph_len:
                                context = [sentences_list[first_sentence_num - 1]] + context
                                cur_sent_len += first_sentence_len
                                first_sentence_num -= 1
                                added = True
                        if not added:
                            break
                    context = " ".join(context)

            log.debug(f"rank, context: {context}")
            contexts.append(context)

        scores_list = self.entity_ranker(contexts, cand_ent_list, cand_ent_descr_list)
        for (entity_substr, candidate_entities, substr_len, entities_scores, scores,) in zip(
                entity_substr_list,
                cand_ent_list,
                substr_lens,
                entities_scores_list,
                scores_list,
        ):
            log.debug(f"len candidate entities {len(candidate_entities)}")
            entities_with_scores = [
                (
                    entity,
                    round(entities_scores.get(entity, (0.0, 0))[0], 2),
                    entities_scores.get(entity, (0.0, 0))[1],
                    round(float(score), 2),
                )
                for entity, score in scores
            ]
            log.debug(f"len entities with scores {len(entities_with_scores)}")
            entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            log.debug(f"--- entities_with_scores {entities_with_scores}")

            if not entities_with_scores:
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            elif entities_with_scores and substr_len == 1 and entities_with_scores[0][1] < 1.0:
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            elif entities_with_scores and (
                    entities_with_scores[0][1] < 0.3
                    or (entities_with_scores[0][3] < 0.13 and entities_with_scores[0][2] < 20)
                    or (entities_with_scores[0][3] < 0.3 and entities_with_scores[0][2] < 4)
                    or entities_with_scores[0][1] < 0.6
            ):
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            else:
                top_entities = [score[0] for score in entities_with_scores]
                top_conf = [score[1:] for score in entities_with_scores]

            log.debug(f"--- top_entities {top_entities} top_conf {top_conf}")

            high_conf_entities = []
            high_conf_nums = []
            for elem_num, (entity, conf) in enumerate(zip(top_entities, top_conf)):
                if len(conf) == 3 and conf[0] == 1.0 and conf[1] > 50 and conf[2] > 0.3:
                    new_conf = list(conf)
                    if new_conf[1] > 55:
                        new_conf[2] = 1.0
                    new_conf = tuple(new_conf)
                    high_conf_entities.append((entity,) + new_conf)
                    high_conf_nums.append(elem_num)

            high_conf_entities = sorted(high_conf_entities, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            for n, elem_num in enumerate(high_conf_nums):
                if 0 <= elem_num - n < len(top_entities):
                    del top_entities[elem_num - n]
                    del top_conf[elem_num - n]

            top_entities = [elem[0] for elem in high_conf_entities] + top_entities
            top_conf = [elem[1:] for elem in high_conf_entities] + top_conf

            log.debug(f"top entities {top_entities} top_conf {top_conf}")

            if self.num_entities_to_return == 1 and top_entities:
                entity_ids_list.append(top_entities[0])
                conf_list.append(top_conf[0])
            else:
                entity_ids_list.append(top_entities[: self.num_entities_to_return])
                conf_list.append(top_conf[: self.num_entities_to_return])
        return entity_ids_list, conf_list
