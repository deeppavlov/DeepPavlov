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
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pymorphy2
import faiss
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
                 word_to_idlist_filename: str,
                 entities_list_filename: str,
                 entities_ranking_filename: str,
                 vectorizer_filename: str,
                 faiss_index_filename: str,
                 chunker: NerChunker = None,
                 ner: Chainer = None,
                 ner_parser: EntityDetectionParser = None,
                 entity_ranker: RelRankerBertInfer = None,
                 num_faiss_candidate_entities: int = 20,
                 num_entities_for_bert_ranking: int = 50,
                 num_faiss_cells: int = 50,
                 use_gpu: bool = True,
                 save_path: str = None,
                 fit_vectorizer: bool = False,
                 max_tfidf_features: int = 1000,
                 include_mention: bool = False,
                 ngram_range: List[int] = None,
                 num_entities_to_return: int = 10,
                 lang: str = "ru",
                 use_descriptions: bool = True,
                 lemmatize: bool = False,
                 **kwargs) -> None:
        """

        Args:
            load_path: path to folder with inverted index files
            word_to_idlist_filename: file with dict of words (keys) and start and end indices in
                entities_list filename of the corresponding entity ids
            entities_list_filename: file with the list of entity ids from the knowledge base
            entities_ranking_filename: file with dict of entity ids (keys) and number of relations in Wikidata
                for entities
            vectorizer_filename: filename with TfidfVectorizer data
            faiss_index_filename: file with Faiss index of words
            chunker: component deeppavlov.models.kbqa.ner_chunker
            ner: config for entity detection
            ner_parser: component deeppavlov.models.kbqa.entity_detection_parser
            entity_ranker: component deeppavlov.models.kbqa.rel_ranking_bert_infer
            num_faiss_candidate_entities: number of nearest neighbors for the entity substring from the text
            num_entities_for_bert_ranking: number of candidate entities for BERT ranking using description and context
            num_faiss_cells: number of Voronoi cells for Faiss index
            use_gpu: whether to use GPU for faster search of candidate entities
            save_path: path to folder with inverted index files
            fit_vectorizer: whether to build index with Faiss library
            max_tfidf_features: maximal number of features for TfidfVectorizer
            include_mention: whether to leave entity mention in the context (during BERT ranking)
            ngram_range: char ngrams range for TfidfVectorizer
            num_entities_to_return: number of candidate entities for the substring which are returned
            lang: russian or english
            use_description: whether to perform entity ranking by context and description
            lemmatize: whether to lemmatize tokens
            **kwargs:
        """
        super().__init__(save_path=save_path, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.word_to_idlist_filename = word_to_idlist_filename
        self.entities_list_filename = entities_list_filename
        self.entities_ranking_filename = entities_ranking_filename
        self.vectorizer_filename = vectorizer_filename
        self.faiss_index_filename = faiss_index_filename
        self.num_entities_for_bert_ranking = num_entities_for_bert_ranking
        self.num_faiss_candidate_entities = num_faiss_candidate_entities
        self.num_faiss_cells = num_faiss_cells
        self.use_gpu = use_gpu
        self.chunker = chunker
        self.ner = ner
        self.ner_parser = ner_parser
        self.entity_ranker = entity_ranker
        self.fit_vectorizer = fit_vectorizer
        self.max_tfidf_features = max_tfidf_features
        self.include_mention = include_mention
        self.ngram_range = ngram_range
        self.num_entities_to_return = num_entities_to_return
        self.lang_str = f"@{lang}"
        if self.lang_str == "@en":
            self.stopwords = set(stopwords.words("english"))
        elif self.lang_str == "@ru":
            self.stopwords = set(stopwords.words("russian"))
        self.use_descriptions = use_descriptions

        self.load()

        if self.fit_vectorizer:
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(self.ngram_range),
                                              max_features=self.max_tfidf_features, max_df=0.85)
            self.vectorizer.fit(self.word_list)
            self.matrix = self.vectorizer.transform(self.word_list)
            self.dense_matrix = self.matrix.toarray()
            if self.num_faiss_cells > 1:
                quantizer = faiss.IndexFlatIP(self.max_tfidf_features)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.max_tfidf_features, self.num_faiss_cells)
                self.faiss_index.train(self.dense_matrix.astype(np.float32))
            else:
                self.faiss_index = faiss.IndexFlatIP(self.max_tfidf_features)
                if self.use_gpu:
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            self.faiss_index.add(self.dense_matrix.astype(np.float32))
            self.save_vectorizers_data()

    def load(self) -> None:
        self.word_to_idlist = load_pickle(self.load_path / self.word_to_idlist_filename)
        self.entities_list = load_pickle(self.load_path / self.entities_list_filename)
        self.word_list = list(self.word_to_idlist.keys())
        self.entities_ranking_dict = load_pickle(self.load_path / self.entities_ranking_filename)
        if not self.fit_vectorizer:
            self.vectorizer = load_pickle(self.load_path / self.vectorizer_filename)
            self.faiss_index = faiss.read_index(str(expand_path(self.faiss_index_filename)))
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

    def save(self) -> None:
        pass

    def save_vectorizers_data(self) -> None:
        save_pickle(self.vectorizer, self.save_path / self.vectorizer_filename)
        faiss.write_index(self.faiss_index, str(expand_path(self.faiss_index_filename)))

    def __call__(self, docs_batch: List[str]):
        """

        Args:
            docs_batch: batch of documents
        Returns:
            batch of lists of candidate entity ids
        """
        text_batch_list, nums_batch_list = self.chunker(docs_batch)
        entity_ids_batch_list = []
        entity_substr_batch_list = []
        entity_positions_batch_list = []
        text_len_batch_list = []
        for text_batch in text_batch_list:
            entity_ids_batch = []
            ner_tokens_batch, ner_probas_batch = self.ner(text_batch)
            entity_substr_batch, _, entity_positions_batch = self.ner_parser(ner_tokens_batch, ner_probas_batch)
            log.debug(f"entity_substr_batch {entity_substr_batch}")
            log.debug(f"entity_positions_batch {entity_positions_batch}")
            entity_substr_batch = [[entity_substr.lower() for tag, entity_substr_list in entity_substr_dict.items()
                                    for entity_substr in entity_substr_list]
                                   for entity_substr_dict in entity_substr_batch]
            entity_positions_batch = [[entity_positions for tag, entity_positions_list in entity_positions_dict.items()
                                       for entity_positions in entity_positions_list]
                                      for entity_positions_dict in entity_positions_batch]
            log.debug(f"entity_substr_batch {entity_substr_batch}")
            log.debug(f"entity_positions_batch {entity_positions_batch}")
            for entity_substr_list, entity_positions_list, context_tokens in \
                    zip(entity_substr_batch, entity_positions_batch, ner_tokens_batch):
                entity_ids_list = []
                if entity_substr_list:
                    entity_ids_list = self.link_entities(entity_substr_list, entity_positions_list, context_tokens)
                entity_ids_batch.append(entity_ids_list)
            entity_ids_batch_list.append(entity_ids_batch)
            entity_substr_batch_list.append(entity_substr_batch)
            entity_positions_batch_list.append(entity_positions_batch)
            text_len_batch_list.append([len(text) for text in ner_tokens_batch])

        doc_entity_ids_batch = []
        doc_entity_substr_batch = []
        doc_entity_positions_batch = []
        doc_entity_ids = []
        doc_entity_substr = []
        doc_entity_positions = []
        cur_doc_num = 0
        text_len_sum = 0
        for entity_ids_batch, entity_substr_batch, entity_positions_batch, text_len_batch, nums_batch in \
                zip(entity_ids_batch_list, entity_substr_batch_list, entity_positions_batch_list,
                    text_len_batch_list, nums_batch_list):
            for entity_ids, entity_substr, entity_positions, text_len, doc_num in \
                    zip(entity_ids_batch, entity_substr_batch, entity_positions_batch, text_len_batch, nums_batch):
                if doc_num == cur_doc_num:
                    doc_entity_ids += entity_ids
                    doc_entity_substr += entity_substr
                    doc_entity_positions += [[pos + text_len_sum for pos in entity_position]
                                             for entity_position in entity_positions]
                    text_len_sum += text_len
                else:
                    doc_entity_ids_batch.append(doc_entity_ids)
                    doc_entity_substr_batch.append(doc_entity_substr)
                    doc_entity_positions_batch.append(doc_entity_positions)
                    doc_entity_ids = entity_ids
                    doc_entity_substr = entity_substr
                    doc_entity_positions = entity_positions
                    cur_doc_num = doc_num
                    text_len_sum = 0
        doc_entity_ids_batch.append(doc_entity_ids)
        doc_entity_substr_batch.append(doc_entity_substr)
        doc_entity_positions_batch.append(doc_entity_positions)

        return doc_entity_substr_batch, doc_entity_positions_batch, doc_entity_ids_batch

    def link_entities(self, entity_substr_list: List[str], entity_positions_list: List[List[int]] = None,
                      context_tokens: List[str] = None) -> List[List[str]]:
        log.debug(f"context_tokens {context_tokens}")
        log.debug(f"entity substr list {entity_substr_list}")
        log.debug(f"entity positions list {entity_positions_list}")
        entity_ids_list = []
        if entity_substr_list:
            entity_substr_list = [[word for word in entity_substr.split(' ')
                                   if word not in self.stopwords and len(word) > 0]
                                  for entity_substr in entity_substr_list]
            words_and_indices = [(self.morph_parse(word), i) for i, entity_substr in enumerate(entity_substr_list)
                                 for word in entity_substr]
            substr_lens = [len(entity_substr) for entity_substr in entity_substr_list]
            log.debug(f"words and indices {words_and_indices}")
            words, indices = zip(*words_and_indices)
            words = list(words)
            indices = list(indices)
            log.debug(f"words {words}")
            log.debug(f"indices {indices}")
            ent_substr_tfidfs = self.vectorizer.transform(words).toarray().astype(np.float32)
            D, I = self.faiss_index.search(ent_substr_tfidfs, self.num_faiss_candidate_entities)
            candidate_entities_dict = defaultdict(list)
            for ind_list, scores_list, index in zip(I, D, indices):
                if self.num_faiss_cells > 1:
                    scores_list = [1.0 - score for score in scores_list]
                candidate_entities = {}
                for ind, score in zip(ind_list, scores_list):
                    start_ind, end_ind = self.word_to_idlist[self.word_list[ind]]
                    for entity in self.entities_list[start_ind:end_ind]:
                        if entity in candidate_entities:
                            if score > candidate_entities[entity]:
                                candidate_entities[entity] = score
                        else:
                            candidate_entities[entity] = score
                candidate_entities_dict[index] += [(entity, cand_entity_len, score)
                                                   for (entity, cand_entity_len), score in candidate_entities.items()]
                log.debug(f"{index} candidate_entities {[self.word_list[ind] for ind in ind_list[:10]]}")
            candidate_entities_total = list(candidate_entities_dict.values())
            candidate_entities_total = [self.sum_scores(candidate_entities, substr_len)
                                        for candidate_entities, substr_len in
                                        zip(candidate_entities_total, substr_lens)]
            log.debug(f"length candidate entities list {len(candidate_entities_total)}")
            candidate_entities_list = []
            entities_scores_list = []
            for candidate_entities in candidate_entities_total:
                log.debug(f"candidate_entities before ranking {candidate_entities[:10]}")
                candidate_entities = [candidate_entity + (self.entities_ranking_dict.get(candidate_entity[0], 0),)
                                      for candidate_entity in candidate_entities]
                candidate_entities_str = '\n'.join([str(candidate_entity) for candidate_entity in candidate_entities])
                candidate_entities = sorted(candidate_entities, key=lambda x: (x[1], x[2]), reverse=True)
                log.debug(f"candidate_entities {candidate_entities[:10]}")
                entities_scores = {entity: (substr_score, pop_score)
                                   for entity, substr_score, pop_score in candidate_entities}
                candidate_entities = [candidate_entity[0] for candidate_entity
                                      in candidate_entities][:self.num_entities_for_bert_ranking]
                log.debug(f"candidate_entities {candidate_entities[:10]}")
                candidate_entities_list.append(candidate_entities)
                if self.num_entities_to_return == 1:
                    entity_ids_list.append(candidate_entities[0])
                else:
                    entity_ids_list.append(candidate_entities[:self.num_entities_to_return])
                entities_scores_list.append(entities_scores)
            if self.use_descriptions:
                entity_ids_list = self.rank_by_description(entity_positions_list, candidate_entities_list,
                                                           entities_scores_list, context_tokens)

        return entity_ids_list

    def morph_parse(self, word):
        morph_parse_tok = self.morph.parse(word)[0]
        normal_form = morph_parse_tok.normal_form
        return normal_form

    def sum_scores(self, candidate_entities: List[Tuple[str, int]], substr_len: int) -> List[Tuple[str, float]]:
        entities_with_scores_sum = defaultdict(int)
        for entity in candidate_entities:
            entities_with_scores_sum[(entity[0], entity[1])] += entity[2]

        entities_with_scores = {}
        for (entity, cand_entity_len), scores_sum in entities_with_scores_sum.items():
            score = min(scores_sum, cand_entity_len) / max(substr_len, cand_entity_len)
            if entity in entities_with_scores:
                if score > entities_with_scores[entity]:
                    entities_with_scores[entity] = score
            else:
                entities_with_scores[entity] = score
        entities_with_scores = list(entities_with_scores.items())

        return entities_with_scores

    def rank_by_description(self, entity_positions_list: List[List[int]],
                            candidate_entities_list: List[List[str]],
                            entities_scores_list: List[Dict[str, Tuple[int, float]]],
                            context_tokens: List[str]) -> List[List[str]]:
        entity_ids_list = []
        for entity_pos, candidate_entities, entities_scores in zip(entity_positions_list, candidate_entities_list,
                                                                   entities_scores_list):
            log.debug(f"entity_pos {entity_pos}")
            log.debug(f"candidate_entities {candidate_entities[:10]}")
            if self.include_mention:
                context = ' '.join(context_tokens[:entity_pos[0]] + ["[ENT]"] +
                                   context_tokens[entity_pos[0]:entity_pos[-1] + 1] + ["[ENT]"] +
                                   context_tokens[entity_pos[-1] + 1:])
            else:
                context = ' '.join(context_tokens[:entity_pos[0]] + ["[ENT]"] + context_tokens[entity_pos[-1] + 1:])
            log.debug(f"context {context}")
            log.debug(f"len candidate entities {len(candidate_entities)}")
            scores = self.entity_ranker.rank_rels(context, candidate_entities)
            entities_with_scores = [(entity, round(entities_scores[entity][0], 2), entities_scores[entity][1],
                                     round(score, 2)) for entity, score in scores]
            log.debug(f"len entities with scores {len(entities_with_scores)}")
            entities_with_scores = [entity for entity in entities_with_scores if entity[3] > 0.1]
            entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            log.debug(f"entities_with_scores {entities_with_scores}")
            top_entities = [score[0] for score in entities_with_scores]
            if self.num_entities_to_return == 1:
                entity_ids_list.append(top_entities[0])
            else:
                entity_ids_list.append(top_entities[:self.num_entities_to_return])
        return entity_ids_list
