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

import nltk
import pymorphy2
from nltk.corpus import stopwords
from rapidfuzz import fuzz

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)
nltk.download("stopwords")


@register("entity_linker")
class EntityLinker(Component, Serializable):
    """
    Class for linking of entity substrings in the document to entities in Wikidata
    """

    def __init__(
            self,
            load_path: str,
            entity_ranker=None,
            entities_database_filename: str = None,
            num_entities_for_bert_ranking: int = 50,
            num_entities_to_return: int = 10,
            max_text_len: int = 300,
            max_paragraph_len: int = 150,
            lang: str = "ru",
            use_descriptions: bool = True,
            use_tags: bool = False,
            lemmatize: bool = False,
            full_paragraph: bool = False,
            use_connections: bool = False,
            **kwargs,
    ) -> None:
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
        self.num_entities_for_bert_ranking = num_entities_for_bert_ranking
        self.entity_ranker = entity_ranker
        self.entities_database_filename = entities_database_filename
        self.num_entities_to_return = num_entities_to_return
        self.max_text_len = max_text_len
        self.max_paragraph_len = max_paragraph_len
        self.lang = f"@{lang}"
        if self.lang == "@en":
            self.stopwords = set(stopwords.words("english"))
        elif self.lang == "@ru":
            self.stopwords = set(stopwords.words("russian"))
        self.use_descriptions = use_descriptions
        self.use_connections = use_connections
        self.use_tags = use_tags
        self.full_paragraph = full_paragraph
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.not_found_str = "not in wiki"
        self.related_tags = {
            "loc": ["gpe", "country", "city", "us_state", "river"],
            "gpe": ["loc", "country", "city", "us_state"],
            "work_of_art": ["product", "law"],
            "product": ["work_of_art"],
            "law": ["work_of_art"],
            "org": ["fac", "business"],
            "business": ["org"]
        }
        self.word_searcher = None
        self.load()

    def load(self) -> None:
        self.conn = sqlite3.connect(str(self.load_path / self.entities_database_filename))
        self.cur = self.conn.cursor()

    def save(self) -> None:
        pass

    def __call__(
            self,
            entity_substr_batch: List[List[str]],
            entity_tags_batch: List[List[str]] = None,
            sentences_batch: List[List[str]] = None,
            entity_offsets_batch: List[List[List[int]]] = None,
            sentences_offsets_batch: List[List[Tuple[int, int]]] = None,
    ):
        if (not sentences_offsets_batch or sentences_offsets_batch[0] is None) and sentences_batch is not None:
            sentences_offsets_batch = []
            for sentences_list in sentences_batch:
                sentences_offsets_list = []
                start = 0
                for sentence in sentences_list:
                    end = start + len(sentence)
                    sentences_offsets_list.append([start, end])
                    start = end + 1
                sentences_offsets_batch.append(sentences_offsets_list)

        if sentences_batch is None:
            sentences_batch = [[] for _ in entity_substr_batch]
            sentences_offsets_batch = [[] for _ in entity_substr_batch]

        log.debug(f"entity_substr: {entity_substr_batch} --- sentences_batch: {sentences_batch} --- "
                  f"entity_offsets: {entity_offsets_batch}")
        if (not entity_offsets_batch or entity_offsets_batch[0] is None) and sentences_batch:
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
        entity_ids_batch, entity_conf_batch, entity_pages_batch, entity_labels_batch = [], [], [], []
        for entity_substr_list, entity_offsets_list, entity_tags_list, sentences_list, sentences_offsets_list in zip(
                entity_substr_batch, entity_offsets_batch, entity_tags_batch, sentences_batch, sentences_offsets_batch
        ):
            entity_ids_list, entity_conf_list, entity_pages_list, entity_labels_list = self.link_entities(
                entity_substr_list,
                entity_offsets_list,
                entity_tags_list,
                sentences_list,
                sentences_offsets_list,
            )
            log.debug(f"entity_ids_list {entity_ids_list} entity_conf_list {entity_conf_list}")
            if self.num_entities_to_return == 1:
                entity_pages_list = [entity_pages[0] for entity_pages in entity_pages_list]
            else:
                entity_pages_list = [entity_pages[: self.num_entities_to_return] for entity_pages in entity_pages_list]
            entity_ids_batch.append(entity_ids_list)
            entity_conf_batch.append(entity_conf_list)
            entity_pages_batch.append(entity_pages_list)
            entity_labels_batch.append(entity_labels_list)
        return entity_ids_batch, entity_conf_batch, entity_pages_batch, entity_labels_batch

    def link_entities(
            self,
            entity_substr_list: List[str],
            entity_offsets_list: List[List[int]],
            entity_tags_list: List[str],
            sentences_list: List[str],
            sentences_offsets_list: List[List[int]],
    ) -> List[List[str]]:
        log.debug(
            f"entity_substr_list {entity_substr_list} entity_tags_list {entity_tags_list} "
            f"entity_offsets_list {entity_offsets_list}"
        )
        entity_ids_list, conf_list, pages_list, entity_label_list, descr_list = [], [], [], [], []
        if entity_substr_list:
            entities_scores_list = []
            cand_ent_scores_list = []
            for entity_substr, tags in zip(entity_substr_list, entity_tags_list):
                for old_symb, new_symb in [("'s", ""), ("@", ""), ("  ", " "), (".", ""), (",", ""), ("-", " "),
                                           ("'", " "), ("!", ""), (":", ""), ("&", ""), ("/", " "), ("  ", " ")]:
                    entity_substr = entity_substr.replace(old_symb, new_symb)
                cand_ent_init = defaultdict(set)
                if len(entity_substr) > 1:
                    if isinstance(tags, str):
                        tags = [tags]
                    tags = [tag.lower() for tag in tags]
                    if tags and not isinstance(tags[0], (list, tuple)):
                        tags = [(tag, 1.0) for tag in tags]
                    cand_ent_init = self.find_exact_match(entity_substr, tags, use_tags=True)
                    new_entity_substr = re.sub(r"\b([a-z]{1}) ([a-z]{1})\b", r"\1\2", entity_substr)
                    if entity_substr != new_entity_substr:
                        new_cand_ent_init = self.find_exact_match(new_entity_substr, tags, use_tags=True)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)

                    init_entity_substr_split = entity_substr.lower().split(" ")
                    if tags[0][0] in {"person", "work_of_art"}:
                        entity_substr_split = [word for word in entity_substr.lower().split(" ") if len(word) > 0]
                    else:
                        entity_substr_split = [
                            word for word in entity_substr.lower().split(" ")
                            if word not in self.stopwords and len(word) > 0
                        ]
                    entity_substr_lemm = ""
                    entity_substr_split_lemm = []
                    if self.lang == "@ru":
                        entity_substr_split_lemm = self.morph_parse(entity_substr_split)
                        entity_substr_lemm = " ".join(entity_substr_split_lemm)
                        if entity_substr_split != entity_substr_split_lemm \
                                or (tags[0][0] == "work_of_art"
                                    and len(entity_substr_split) != len(init_entity_substr_split)):
                            new_cand_ent_init = self.find_fuzzy_match(entity_substr_split, tags, use_tags=True)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                        if entity_substr_split != entity_substr_split_lemm:
                            new_cand_ent_init = self.find_exact_match(entity_substr_lemm, tags, use_tags=True)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                            new_cand_ent_init = self.find_fuzzy_match(entity_substr_split_lemm, tags, use_tags=True)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)

                    all_low_conf = self.define_all_low_conf(cand_ent_init, 1.0)
                    clean_tags, corr_tags, corr_clean_tags = self.correct_tags(tags)

                    if (not cand_ent_init or all_low_conf) and corr_tags:
                        corr_cand_ent_init = self.find_exact_match(entity_substr, corr_tags, use_tags=True)
                        cand_ent_init = self.unite_dicts(cand_ent_init, corr_cand_ent_init)
                        if self.lang == "@ru" and entity_substr_split != entity_substr_split_lemm:
                            new_cand_ent_init = self.find_exact_match(entity_substr_lemm, corr_tags, use_tags=True)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                            new_cand_ent_init = self.find_fuzzy_match(entity_substr_split_lemm, corr_tags,
                                                                      use_tags=True)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)

                    if not cand_ent_init and len(entity_substr_split) == 1 and self.word_searcher:
                        corr_words = self.word_searcher(entity_substr_split[0], set(clean_tags + corr_clean_tags))
                        if corr_words:
                            cand_ent_init = self.find_exact_match(corr_words[0], tags + corr_tags, use_tags=True)

                    if not cand_ent_init and len(entity_substr_split) > 1:
                        cand_ent_init = self.find_fuzzy_match(entity_substr_split, tags)

                    all_low_conf = self.define_all_low_conf(cand_ent_init, 0.85)
                    if not cand_ent_init or all_low_conf:
                        new_cand_ent_init = self.find_exact_match(entity_substr, tags, use_tags=False)
                        cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                        if self.lang == "@ru" and entity_substr_split != entity_substr_split_lemm \
                                and (tags[0][0] == "e" or not cand_ent_init):
                            new_cand_ent_init = self.find_fuzzy_match(entity_substr_split, tags, use_tags=False)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)
                            new_cand_ent_init = self.find_fuzzy_match(entity_substr_split_lemm, tags, use_tags=False)
                            cand_ent_init = self.unite_dicts(cand_ent_init, new_cand_ent_init)

                cand_ent_scores = []
                for entity in cand_ent_init:
                    entities_scores = list(cand_ent_init[entity])
                    entities_scores = sorted(entities_scores, key=lambda x: (x[0], x[2], x[1]), reverse=True)
                    cand_ent_scores.append(([entity] + list(entities_scores[0])))

                cand_ent_scores = sorted(cand_ent_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
                cand_ent_scores = cand_ent_scores[: self.num_entities_for_bert_ranking]
                cand_ent_scores_list.append(cand_ent_scores)
                entity_ids = [elem[0] for elem in cand_ent_scores]
                scores = [elem[1:4] for elem in cand_ent_scores]
                entities_scores_list.append(
                    {entity_id: entity_scores for entity_id, entity_scores in zip(entity_ids, scores)}
                )
                entity_ids_list.append(entity_ids)
                pages = [elem[4] for elem in cand_ent_scores]
                entity_labels = [elem[5] for elem in cand_ent_scores]
                pages_list.append({entity_id: page for entity_id, page in zip(entity_ids, pages)})
                entity_label_list.append(
                    {entity_id: entity_label for entity_id, entity_label in zip(entity_ids, entity_labels)})
                descr_list.append([elem[6] for elem in cand_ent_scores])

            if self.use_descriptions:
                substr_lens = [len(entity_substr.split()) for entity_substr in entity_substr_list]
                entity_ids_list, conf_list = self.rank_by_description(
                    entity_substr_list,
                    entity_offsets_list,
                    entity_ids_list,
                    descr_list,
                    entities_scores_list,
                    sentences_list,
                    sentences_offsets_list,
                    substr_lens,
                )
        entity_label_list = [[label_dict.get(entity_id, "") for entity_id in entity_ids]
                             for entity_ids, label_dict in zip(entity_ids_list, entity_label_list)]
        pages_list = [[pages_dict.get(entity_id, "") for entity_id in entity_ids]
                      for entity_ids, pages_dict in zip(entity_ids_list, pages_list)]

        return entity_ids_list, conf_list, pages_list, entity_label_list

    def define_all_low_conf(self, cand_ent_init, thres):
        all_low_conf = True
        for entity_id in cand_ent_init:
            entity_info_set = cand_ent_init[entity_id]
            for entity_info in entity_info_set:
                if entity_info[0] >= thres:
                    all_low_conf = False
                    break
            if not all_low_conf:
                break
        return all_low_conf

    def correct_tags(self, tags):
        clean_tags = [tag for tag, conf in tags]
        corr_tags, corr_clean_tags = [], []
        for tag, conf in tags:
            if tag in self.related_tags:
                corr_tag_list = self.related_tags[tag]
                for corr_tag in corr_tag_list:
                    if corr_tag not in clean_tags and corr_tag not in corr_clean_tags:
                        corr_tags.append([corr_tag, conf])
                        corr_clean_tags.append(corr_tag)
        return clean_tags, corr_tags, corr_clean_tags

    def unite_dicts(self, cand_ent_init, new_cand_ent_init):
        for entity_id in new_cand_ent_init:
            if entity_id in cand_ent_init:
                for entity_info in new_cand_ent_init[entity_id]:
                    cand_ent_init[entity_id].add(entity_info)
            else:
                cand_ent_init[entity_id] = new_cand_ent_init[entity_id]
        return cand_ent_init

    def normal_form(self, word):
        morph_parse_tok = self.morph.parse(word)[0].inflect({"nomn"})
        if morph_parse_tok:
            normal_form = morph_parse_tok.word
        else:
            normal_form = word
        return normal_form

    def morph_parse(self, words):
        words = [self.normal_form(word) for word in words]
        if len(words) > 1 and self.morph.parse(words[-1])[0].tag.POS == "NOUN":
            gender = self.morph.parse(words[-1])[0].tag.gender
            if gender:
                for i in range(len(words) - 1):
                    if self.morph.parse(words[i])[0].tag.POS == "ADJF":
                        words[i] = self.morph.parse(words[i])[0].inflect({gender}).word
        return words

    def process_cand_ent(self, cand_ent_init, entities_and_ids, entity_substr_split, tag, tag_conf, use_tags):
        for entity_title, entity_id, entity_rels, ent_tag, page, entity_label, descr in entities_and_ids:
            if (ent_tag == tag and use_tags) or not use_tags:
                substr_score = self.calc_substr_score(entity_title, entity_substr_split, entity_id, tag, ent_tag)
                cand_ent_init[entity_id].add((substr_score, entity_rels, tag_conf, page, entity_label, descr))
        return cand_ent_init

    def sanitize_substr(self, entity_substr, tag):
        if tag == "person":
            entity_substr_split = entity_substr.split()
            if len(entity_substr_split) > 1 and len(entity_substr_split[-1]) > 1 and len(entity_substr_split[-2]) == 1:
                entity_substr = entity_substr_split[-1]
        return entity_substr

    def find_exact_match(self, entity_substr, tags, use_tags=True):
        entity_substr = entity_substr.lower()
        entity_substr_split = entity_substr.split()
        cand_ent_init = defaultdict(set)
        for tag, tag_conf in tags:
            entity_substr = self.sanitize_substr(entity_substr, tag)
            query = "SELECT * FROM inverted_index WHERE title MATCH '{}';".format(entity_substr)
            entities_and_ids = []
            try:
                res = self.cur.execute(query)
                entities_and_ids = res.fetchall()
            except:
                log.info(f"error in query execute {query}")
            if entities_and_ids:
                cand_ent_init = self.process_cand_ent(
                    cand_ent_init, entities_and_ids, entity_substr_split, tag, tag_conf, use_tags)
        return cand_ent_init

    def find_fuzzy_match(self, entity_substr_split, tags, use_tags=True):
        cand_ent_init = defaultdict(set)
        for tag, tag_conf in tags:
            for word in entity_substr_split:
                if len(word) > 1:
                    query = "SELECT * FROM inverted_index WHERE title MATCH '{}';".format(word)
                    part_entities_and_ids = []
                    try:
                        res = self.cur.execute(query)
                        part_entities_and_ids = res.fetchall()
                    except:
                        log.info(f"error in query execute {query}")
                    if part_entities_and_ids:
                        cand_ent_init = self.process_cand_ent(
                            cand_ent_init, part_entities_and_ids, entity_substr_split, tag, tag_conf, use_tags)
        return cand_ent_init

    def calc_substr_score(self, entity_title, entity_substr_split, entity_id, tag, ent_tag):
        if self.lang == "@ru":
            entity_title = entity_title.replace("ё", "е")
        label_tokens = entity_title.split()
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
                        if fuzz_score >= 75.0 and not found:
                            cnt += fuzz_score * 0.01
                            break
        substr_score = round(cnt / max(len(label_tokens), len(entity_substr_split)), 3)
        if len(label_tokens) == 2 and len(entity_substr_split) == 1:
            if entity_substr_split[0] == label_tokens[1]:
                substr_score = 0.5
            elif entity_substr_split[0] == label_tokens[0]:
                substr_score = 0.3
        if tag == ent_tag and tag.lower() == "person" and len(entity_substr_split) > 1 \
                and len(entity_substr_split[-1]) > 1 and len(entity_substr_split[-2]) == 1 \
                and len(label_tokens) == len(entity_substr_split):
            cnt = 0.0
            for j in range(len(label_tokens) - 1):
                if label_tokens[j][0] == entity_substr_split[j][0]:
                    cnt += 1.0
            if label_tokens[-1] == entity_substr_split[-1]:
                cnt += 1.0
            new_substr_score = cnt / len(label_tokens)
            substr_score = max(substr_score, new_substr_score)
        return substr_score

    def rank_by_description(
            self,
            entity_substr_list: List[str],
            entity_offsets_list: List[List[int]],
            cand_ent_list: List[List[str]],
            cand_ent_descr_list: List[List[str]],
            entities_scores_list: List[Dict[str, Tuple[int, float]]],
            sentences_list: List[str],
            sentences_offsets_list: List[Tuple[int, int]],
            substr_lens: List[int],
    ) -> List[List[str]]:
        entity_ids_list = []
        conf_list = []
        contexts = []
        for entity_start_offset, entity_end_offset in entity_offsets_list:
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
                text_before = sentence[start_of_sentence:rel_start_offset]
                text_after = sentence[rel_end_offset:end_of_sentence]
                context = text_before + "[ent]" + text_after
                if self.full_paragraph:
                    cur_sent_len = len(re.findall(self.re_tokenizer, context))
                    first_sentence_num = found_sentence_num
                    last_sentence_num = found_sentence_num
                    context = [context]
                    while True:
                        added = False
                        if last_sentence_num < len(sentences_list) - 1:
                            sentence_tokens = re.findall(self.re_tokenizer, sentences_list[last_sentence_num + 1])
                            last_sentence_len = len(sentence_tokens)
                            if cur_sent_len + last_sentence_len < self.max_paragraph_len:
                                context.append(sentences_list[last_sentence_num + 1])
                                cur_sent_len += last_sentence_len
                                last_sentence_num += 1
                                added = True
                        if first_sentence_num > 0:
                            sentence_tokens = re.findall(self.re_tokenizer, sentences_list[first_sentence_num - 1])
                            first_sentence_len = len(sentence_tokens)
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

        for context, candidate_entities, substr_len, entities_scores, scores in zip(
                contexts, cand_ent_list, substr_lens, entities_scores_list, scores_list
        ):
            log.debug(f"len candidate entities {len(candidate_entities)}")
            if len(context.split()) < 4:
                entities_with_scores = [
                    (
                        entity,
                        round(entities_scores.get(entity, (0.0, 0))[0], 2),
                        entities_scores.get(entity, (0.0, 0))[1],
                        0.95,
                    )
                    for entity, score in scores
                ]
            else:
                entities_with_scores = [
                    (
                        entity,
                        round(entities_scores.get(entity, (0.0, 0))[0], 2),
                        entities_scores.get(entity, (0.0, 0))[1],
                        round(score, 2),
                    )
                    for entity, score in scores
                ]
            log.debug(f"len entities with scores {len(entities_with_scores)}")
            entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            log.debug(f"--- entities_with_scores {entities_with_scores}")

            if not entities_with_scores:
                top_entities = []
                top_conf = []
            elif entities_with_scores and substr_len == 1 and entities_with_scores[0][1] < 1.0:
                top_entities = []
                top_conf = []
            elif entities_with_scores and (
                    entities_with_scores[0][1] < 0.3
                    or (entities_with_scores[0][3] < 0.13 and entities_with_scores[0][2] < 20)
                    or (entities_with_scores[0][3] < 0.3 and entities_with_scores[0][2] < 4)
                    or entities_with_scores[0][1] < 0.6
            ):
                top_entities = []
                top_conf = []
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
                if elem_num - n >= 0 and elem_num - n < len(top_entities):
                    del top_entities[elem_num - n]
                    del top_conf[elem_num - n]

            log.debug(f"top entities {top_entities} top_conf {top_conf}")
            log.debug(f"high_conf_entities {high_conf_entities}")

            top_entities = [elem[0] for elem in high_conf_entities] + top_entities
            top_conf = [elem[1:] for elem in high_conf_entities] + top_conf

            if not top_entities:
                entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[2], x[3]), reverse=True)
                top_entities = [score[0] for score in entities_with_scores]
                top_conf = [score[1:] for score in entities_with_scores]

            log.debug(f"top entities {top_entities} top_conf {top_conf}")

            if self.num_entities_to_return == 1 and top_entities:
                entity_ids_list.append(top_entities[0])
                conf_list.append(top_conf[0])
            else:
                entity_ids_list.append(top_entities[: self.num_entities_to_return])
                conf_list.append(top_conf[: self.num_entities_to_return])
        return entity_ids_list, conf_list
