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
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pymorphy2
from nltk.corpus import stopwords
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import load_pickle, save_pickle
from deeppavlov.models.kbqa.entity_detection_parser import EntityDetectionParser
from deeppavlov.models.kbqa.rel_ranking_bert_infer import RelRankerBertInfer

log = getLogger(__name__)


@register('ner_chunker')
class NerChunker(Component):
    """
        Class to split documents into chunks of max_chunk_len symbols so that the length will not exceed
        maximal sequence length to feed into BERT
    """

    def __init__(self, max_chunk_len: int = 300, batch_size: int = 30, **kwargs):
        """

        Args:
            max_chunk_len: maximal length of chunks into which the document is split
            batch_size: how many chunks are in batch
        """
        self.max_chunk_len = max_chunk_len
        self.batch_size = batch_size

    def __call__(self, docs_batch: List[str]) -> Tuple[List[List[str]], List[List[int]]]:
        """
        This method splits each document in the batch into chunks wuth the maximal length of max_chunk_len

        Args:
            docs_batch: batch of documents

        Returns:
            batch of lists of document chunks for each document
            batch of lists of numbers of documents which correspond to chunks
        """
        text_batch_list = []
        text_batch = []
        nums_batch_list = []
        nums_batch = []
        count_texts = 0
        text = ""
        curr_doc = 0
        for n, doc in enumerate(docs_batch):
            sentences = sent_tokenize(doc)
            for sentence in sentences:
                if len(text) + len(sentence) < self.max_chunk_len and n == curr_doc:
                    text += f"{sentence} "
                else:
                    if count_texts < self.batch_size:
                        text_batch.append(text.strip())
                        if n == curr_doc:
                            nums_batch.append(n)
                        else:
                            nums_batch.append(n - 1)
                        count_texts += 1
                    else:
                        text_batch_list.append(text_batch)
                        text_batch = []
                        nums_batch_list.append(nums_batch)
                        nums_batch = [n]
                        count_texts = 0
                    curr_doc = n
                    text = f"{sentence} "

        if text:
            text_batch.append(text.strip())
            text_batch_list.append(text_batch)
            nums_batch.append(len(docs_batch) - 1)
            nums_batch_list.append(nums_batch)

        return text_batch_list, nums_batch_list


@register('entity_linker')
class EntityLinker(Component, Serializable):
    """
        Class for linking of entity substrings in the document to entities in Wikidata
    """

    def __init__(self, load_path: str,
                 entities_database_filename: str,
                 entity_ranker = None,
                 num_entities_for_bert_ranking: int = 50,
                 ngram_range: List[int] = None,
                 num_entities_to_return: int = 10,
                 max_text_len: int = 300,
                 lang: str = "ru",
                 use_descriptions: bool = True,
                 lemmatize: bool = False,
                 full_paragraph: bool = False,
                 **kwargs) -> None:
        """

        Args:
            load_path: path to folder with inverted index files
            entity_ranker: component deeppavlov.models.kbqa.rel_ranking_bert
            num_entities_for_bert_ranking: number of candidate entities for BERT ranking using description and context
            ngram_range: char ngrams range for TfidfVectorizer
            num_entities_to_return: number of candidate entities for the substring which are returned
            lang: russian or english
            use_description: whether to perform entity ranking by context and description
            lemmatize: whether to lemmatize tokens
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.entities_database_filename = entities_database_filename
        self.num_entities_for_bert_ranking = num_entities_for_bert_ranking
        self.entity_ranker = entity_ranker
        self.num_entities_to_return = num_entities_to_return
        self.max_text_len = max_text_len
        self.lang_str = f"@{lang}"
        if self.lang_str == "@en":
            self.stopwords = set(stopwords.words("english"))
        elif self.lang_str == "@ru":
            self.stopwords = set(stopwords.words("russian"))
        self.use_descriptions = use_descriptions
        self.full_paragraph = full_paragraph
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")

        self.load()

    def load(self) -> None:
        self.conn = sqlite3.connect(str(self.load_path / self.entities_database_filename))
        self.cur = self.conn.cursor()

    def save(self) -> None:
        pass

    def __call__(self, entity_substr_batch: List[List[str]],
                       entity_offsets_batch: List[List[List[int]]],
                       sentences_batch: List[List[str]],
                       sentences_offsets_batch: List[List[Tuple[int, int]]] = None):
        if sentences_offsets_batch is None:
            sentences_offsets_batch = []
            for sentences_list in sentences_batch:
                sentences_offsets_list = []
                start = 0
                for sentence in sentences_list:
                    end = start + len(sentence)
                    sentences_offsets_list.append([start, end])
                    start = end + 1
                sentences_offsets_batch.append(sentences_offsets_list)
        
        entity_ids_batch = []
        for entity_substr_list, entity_offsets_list, sentences_list, sentences_offsets_list in \
                zip(entity_substr_batch, entity_offsets_batch, sentences_batch, sentences_offsets_batch):
            entity_ids_list = self.link_entities(entity_substr_list, entity_offsets_list, sentences_list,
                                                 sentences_offsets_list)
            entity_ids_batch.append(entity_ids_list)
        return entity_ids_batch

    def link_entities(self, entity_substr_list: List[str], entity_offsets_list: List[List[int]],
                            sentences_list: List[str], sentences_offsets_list: List[List[int]]) -> List[List[str]]:
        entity_ids_list = []
        if entity_substr_list:
            entities_scores_list = []
            entity_substr_split_list = [[word for word in entity_substr.split(' ')
                                   if word not in self.stopwords and len(word) > 0]
                                  for entity_substr in entity_substr_list]
            for entity_substr, entity_substr_split in zip(entity_substr_list, entity_substr_split_list):
                cand_ent_init = defaultdict(set)
                res = self.cur.execute("SELECT * FROM inverted_index WHERE title MATCH '{}';".format(entity_substr))
                entities_and_ids = res.fetchall()
                if entities_and_ids:
                    for cand_entity_title, cand_entity_id, cand_entity_rels in entities_and_ids:
                        cand_ent_init[cand_entity_id].add((1.0, cand_entity_rels))
                else:
                    for word in entity_substr_split:
                        res = self.cur.execute("SELECT * FROM inverted_index WHERE title MATCH '{}';".format(word))
                        part_entities_and_ids = res.fetchall()
                        for cand_entity_title, cand_entity_id, cand_entity_rels in part_entities_and_ids:
                            substr_score = self.calc_substr_score(cand_entity_title, entity_substr_split)
                            cand_ent_init[cand_entity_id].add((substr_score, cand_entity_rels))
                cand_ent_scores = []
                for entity in cand_ent_init:
                    entities_scores = list(cand_ent_init[entity])
                    entities_scores = sorted(entities_scores, key=lambda x: (x[0], x[1]), reverse=True)
                    cand_ent_scores.append((entity, entities_scores[0]))
                
                cand_ent_scores = sorted(cand_ent_scores, key=lambda x: (x[1][0], x[1][1]), reverse=True)
                cand_ent_scores = cand_ent_scores[:self.num_entities_for_bert_ranking]
                entity_ids = [elem[0] for elem in cand_ent_scores]
                entities_scores_list.append({ent: score for ent, score in cand_ent_scores})
                entity_ids_list.append(entity_ids)
            
            if self.use_descriptions:
                entity_descr_list = []
                for entity_ids in entity_ids_list:
                    entity_descrs = []
                    for entity_id in entity_ids:
                        res = self.cur.execute("SELECT * FROM entity_labels WHERE entity='{}';".format(entity_id))
                        entity_info = res.fetchall()
                        if entity_info:
                            entity_descrs.append(entity_info[0][2])
                        else:
                            entity_descrs.append("")
                    entity_descr_list.append(entity_descrs)
                substr_lens = [len(entity_substr.split()) for entity_substr in entity_substr_list]
                entity_ids_list, conf_list = self.rank_by_description(entity_substr_list, entity_offsets_list,
                                                                      entity_ids_list, entity_descr_list,
                                                                      entities_scores_list, sentences_list,
                                                                      sentences_offsets_list, substr_lens)
        return entity_ids_list

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
                            found = True
                            break
        substr_score = round(cnt / max(len(label_tokens), len(entity_substr_split)), 3)
        return substr_score
    
    def rank_by_description(self, entity_substr_list: List[str],
                                  entity_offsets_list: List[List[int]],
                                  cand_ent_list: List[List[str]],
                                  cand_ent_descr_list: List[List[str]],
                                  entities_scores_list: List[Dict[str, Tuple[int, float]]],
                                  sentences_list: List[str],
                                  sentences_offsets_list: List[Tuple[int, int]],
                                  substr_lens: List[int]) -> List[List[str]]:
        entity_ids_list = []
        conf_list = []
        contexts = []
        for entity_substr, (entity_start_offset, entity_end_offset), candidate_entities in \
                zip(entity_substr_list, entity_offsets_list, cand_ent_list):
            sentence = ""
            rel_start_offset = 0
            rel_end_offset = 0
            found_sentence_num = 0
            for num, (sent, (sent_start_offset, sent_end_offset)) in \
                    enumerate(zip(sentences_list, sentences_offsets_list)):
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
                context = sentence[start_of_sentence:rel_start_offset] + "[ENT]" + \
                          sentence[rel_end_offset:end_of_sentence]
                if self.full_paragraph:
                    cur_sent_len = len(re.findall(self.re_tokenizer, context))
                    first_sentence_num = found_sentence_num
                    last_sentence_num = found_sentence_num
                    context = [context]
                    while True:
                        added = False
                        if last_sentence_num < len(sentences_list) - 1:
                            last_sentence_len = len(
                                re.findall(self.re_tokenizer, sentences_list[last_sentence_num + 1]))
                            if cur_sent_len + last_sentence_len < self.max_paragraph_len:
                                context.append(sentences_list[last_sentence_num + 1])
                                cur_sent_len += last_sentence_len
                                last_sentence_num += 1
                                added = True
                        if first_sentence_num > 0:
                            first_sentence_len = len(
                                re.findall(self.re_tokenizer, sentences_list[first_sentence_num - 1]))
                            if cur_sent_len + first_sentence_len < self.max_paragraph_len:
                                context = [sentences_list[first_sentence_num - 1]] + context
                                cur_sent_len += first_sentence_len
                                first_sentence_num -= 1
                                added = True
                        if not added:
                            break
                    context = ' '.join(context)

            log.info(f"rank, context: {context}")
            contexts.append(context)

        scores_list = self.entity_ranker(contexts, cand_ent_list, cand_ent_descr_list)

        for entity_substr, candidate_entities, substr_len, entities_scores, scores in \
                zip(entity_substr_list, cand_ent_list, substr_lens, entities_scores_list, scores_list):
            log.info(f"len candidate entities {len(candidate_entities)}")
            entities_with_scores = [(entity, round(entities_scores.get(entity, (0.0, 0))[0], 2),
                                     entities_scores.get(entity, (0.0, 0))[1],
                                     round(score, 2)) for entity, score in scores]
            log.info(f"len entities with scores {len(entities_with_scores)}")
            entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            log.info(f"entities_with_scores {entities_with_scores}")

            if not entities_with_scores:
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            elif entities_with_scores and substr_len == 1 and entities_with_scores[0][1] < 1.0:
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            elif entities_with_scores and (entities_with_scores[0][1] < 0.3
                                           or (entities_with_scores[0][3] < 0.13 and entities_with_scores[0][2] < 20)
                                           or (entities_with_scores[0][3] < 0.3 and entities_with_scores[0][2] < 4)
                                           or entities_with_scores[0][1] < 0.6):
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            else:
                top_entities = [score[0] for score in entities_with_scores]
                top_conf = [score[1:] for score in entities_with_scores]

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
                if elem_num - n >= 0 and elem_num - n < len(top_entities):
                    del top_entities[elem_num - n]

            top_entities = [elem[0] for elem in high_conf_entities] + top_entities
            top_conf = [elem[1:] for elem in high_conf_entities] + top_conf
            
            if self.num_entities_to_return == 1 and top_entities:
                entity_ids_list.append(top_entities[0])
                conf_list.append(top_conf[0])
            else:
                entity_ids_list.append(top_entities[:self.num_entities_to_return])
                conf_list.append(top_conf[:self.num_entities_to_return])
        return entity_ids_list, conf_list
