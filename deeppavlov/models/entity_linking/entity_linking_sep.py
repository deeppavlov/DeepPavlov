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
import time
from logging import getLogger
from typing import List, Dict, Tuple
from collections import defaultdict
from string import punctuation

import numpy as np
import pymorphy2
import faiss
import fasttext
from nltk.corpus import stopwords
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import load_pickle, save_pickle
from deeppavlov.models.kbqa.entity_detection_parser import EntityDetectionParser

log = getLogger(__name__)


@register('ner_chunk_model')
class NerChunkModel(Component):
    """
        Class for linking of entity substrings in the document to entities in Wikidata
    """

    def __init__(self, ner: Chainer,
                 ner_parser: EntityDetectionParser,
                 **kwargs) -> None:
        """

        Args:
            ner: config for entity detection
            ner_parser: component deeppavlov.models.kbqa.entity_detection_parser
            **kwargs:
        """
        self.ner = ner
        self.ner_parser = ner_parser

    def __call__(self, text_batch_list: List[List[str]],
                 nums_batch_list: List[List[int]],
                 sentences_offsets_batch_list: List[List[List[Tuple[int, int]]]],
                 sentences_batch_list: List[List[List[str]]]
                 ):
        """

        Args:
            text_batch_list: list of document chunks
            nums_batch_list: nums of documents
            sentences_offsets_batch_list: indices of start and end symbols of sentences in text
            sentences_batch_list: list of sentences from texts
        Returns:
            doc_entity_substr_batch: entity substrings
            doc_entity_offsets_batch: indices of start and end symbols of entities in text
            doc_tags_batch: entity tags (PER, LOC, ORG)
            doc_sentences_offsets_batch: indices of start and end symbols of sentences in text
            doc_sentences_batch: list of sentences from texts
        
        Examples of input arguments:
            text_batch_list: [['Екатеринбург - город в России, административный центр Уральского федерального 
                                округа и Свердловской области. Расположен на восточном склоне Среднего Урала,
                                по берегам реки Исети.']]
            nums_batch_list: [[0]]
            sentences_offsets_batch_list: [[[(0, 108), (109, 178)]]]
            sentences_batch_list: [[['Екатеринбург - город в России, административный центр Уральского федерального
                                      округа и Свердловской области.',
                                      'Расположен на восточном склоне Среднего Урала, по берегам реки Исети.']]]
        
        Examples of outputs:
            doc_entity_substr_batch: [['екатеринбург', 'россии', 'уральского федерального округа',
                                       'свердловской области', 'среднего урала', 'исети']]
            doc_entity_offsets_batch: [[(0, 12), (23, 29), (54, 84), (87, 107), (140, 154), (172, 177)]]
            doc_tags_batch: [['LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC']]
            doc_sentences_offsets_batch: [[(0, 108), (109, 178)]]
            doc_sentences_batch: [['Екатеринбург - город в России, административный центр Уральского федерального
                                    округа и Свердловской области.',
                                    'Расположен на восточном склоне Среднего Урала, по берегам реки Исети.']]
        """
        entity_substr_batch_list = []
        entity_offsets_batch_list = []
        tags_batch_list = []
        entity_probas_batch_list = []
        text_len_batch_list = []
        for text_batch, sentences_offsets_batch, sentences_batch in \
                zip(text_batch_list, sentences_offsets_batch_list, sentences_batch_list):
            tm_ner_st = time.time()
            ner_tokens_batch, ner_tokens_offsets_batch, ner_probas_batch, probas_batch = self.ner(text_batch)
            entity_substr_batch, entity_positions_batch, entity_probas_batch = \
                self.ner_parser(ner_tokens_batch, ner_probas_batch, probas_batch)
            tm_ner_end = time.time()
            log.debug(f"ner time {tm_ner_end - tm_ner_st}")
            log.debug(f"entity_substr_batch {entity_substr_batch}")
            log.debug(f"entity_positions_batch {entity_positions_batch}")
            entity_pos_tags_probas_batch = [[(entity_substr.lower(), entity_substr_positions, tag, entity_proba)
                                             for tag, entity_substr_list in entity_substr_dict.items()
                                             for entity_substr, entity_substr_positions, entity_proba in
                                             zip(entity_substr_list, entity_positions_dict[tag],
                                                 entity_probas_dict[tag])]
                                            for entity_substr_dict, entity_positions_dict, entity_probas_dict in
                                            zip(entity_substr_batch, entity_positions_batch, entity_probas_batch)]
            entity_substr_batch = []
            entity_offsets_batch = []
            tags_batch = []
            probas_batch = []
            for entity_pos_tags_probas, ner_tokens_offsets_list in \
                    zip(entity_pos_tags_probas_batch, ner_tokens_offsets_batch):
                if entity_pos_tags_probas:
                    entity_offsets_list = []
                    entity_substr_list, entity_positions_list, tags_list, probas_list = zip(*entity_pos_tags_probas)
                    for entity_positions in entity_positions_list:
                        start_offset = ner_tokens_offsets_list[entity_positions[0]][0]
                        end_offset = ner_tokens_offsets_list[entity_positions[-1]][1]
                        entity_offsets_list.append((start_offset, end_offset))
                else:
                    entity_substr_list, entity_offsets_list, tags_list, probas_list = [], [], [], []
                entity_substr_batch.append(list(entity_substr_list))
                entity_offsets_batch.append(list(entity_offsets_list))
                tags_batch.append(list(tags_list))
                probas_batch.append(list(probas_list))

            log.debug(f"entity_substr_batch {entity_substr_batch}")
            log.debug(f"entity_offsets_batch {entity_offsets_batch}")

            entity_substr_batch_list.append(entity_substr_batch)
            tags_batch_list.append(tags_batch)
            entity_offsets_batch_list.append(entity_offsets_batch)
            entity_probas_batch_list.append(probas_batch)
            text_len_batch_list.append([len(text) for text in text_batch])

        doc_entity_substr_batch, doc_tags_batch, doc_entity_offsets_batch, doc_probas_batch = [], [], [], []
        doc_sentences_offsets_batch, doc_sentences_batch = [], []
        doc_entity_substr, doc_tags, doc_probas, doc_entity_offsets = [], [], [], []
        doc_sentences_offsets, doc_sentences = [], []
        cur_doc_num = 0
        text_len_sum = 0
        for entity_substr_batch, tags_batch, probas_batch, entity_offsets_batch, sentences_offsets_batch, \
            sentences_batch, text_len_batch, nums_batch in \
                zip(entity_substr_batch_list, tags_batch_list, entity_probas_batch_list, entity_offsets_batch_list,
                    sentences_offsets_batch_list, sentences_batch_list, text_len_batch_list, nums_batch_list):
            for entity_substr, tag, probas, entity_offsets, sentences_offsets, sentences, text_len, doc_num in \
                    zip(entity_substr_batch, tags_batch, probas_batch, entity_offsets_batch, sentences_offsets_batch,
                        sentences_batch, text_len_batch, nums_batch):
                if doc_num == cur_doc_num:
                    doc_entity_substr += entity_substr
                    doc_tags += tag
                    doc_probas += probas
                    doc_entity_offsets += [(start_offset + text_len_sum, end_offset + text_len_sum)
                                           for start_offset, end_offset in entity_offsets]
                    doc_sentences_offsets += [(start_offset + text_len_sum, end_offset + text_len_sum)
                                              for start_offset, end_offset in sentences_offsets]
                    doc_sentences += sentences
                    text_len_sum += text_len + 1
                else:
                    doc_entity_substr_batch.append(doc_entity_substr)
                    doc_tags_batch.append(doc_tags)
                    doc_probas_batch.append(doc_probas)
                    doc_entity_offsets_batch.append(doc_entity_offsets)
                    doc_sentences_offsets_batch.append(doc_sentences_offsets)
                    doc_sentences_batch.append(doc_sentences)
                    doc_entity_substr = entity_substr
                    doc_tags = tag
                    doc_probas = probas
                    doc_entity_offsets = entity_offsets
                    doc_sentences_offsets = sentences_offsets
                    doc_sentences = sentences
                    cur_doc_num = doc_num
                    text_len_sum = 0
        doc_entity_substr_batch.append(doc_entity_substr)
        doc_entity_substr_batch = self.sanitize_entity_substr(doc_entity_substr_batch)
        doc_tags_batch.append(doc_tags)
        doc_probas_batch.append(doc_probas)
        doc_entity_offsets_batch.append(doc_entity_offsets)
        doc_sentences_offsets_batch.append(doc_sentences_offsets)
        doc_sentences_batch.append(doc_sentences)

        return doc_entity_substr_batch, doc_entity_offsets_batch, doc_tags_batch, \
               doc_sentences_offsets_batch, doc_sentences_batch, doc_probas_batch
               
    def sanitize_entity_substr(self, entity_substr_batch: List[List[str]]):
        new_entity_substr_batch = []
        for entity_substr_list in entity_substr_batch:
            new_entity_substr_list = []
            for entity_substr in entity_substr_list:
                entity_substr = entity_substr.replace(" ' ", ' "').replace(" '", '"').replace("' ", '"')
                new_entity_substr_list.append(entity_substr)
            new_entity_substr_batch.append(new_entity_substr_list)
        return new_entity_substr_batch


@register('entity_linker_sep')
class EntityLinkerSep(Component, Serializable):
    """
        Class for linking of entity substrings in the document to entities in Wikidata
    """

    def __init__(self, load_path: str,
                 word_to_idlist_filename: str,
                 entities_ranking_filename: str,
                 entities_types_sets_filename: str,
                 q_to_label_filename: str,
                 q_to_label_out_filename: str,
                 q_to_descr_filename: str,
                 tfidf_vectorizer_filename: str,
                 tfidf_faiss_index_filename: str,
                 fasttext_vectorizer_filename: str,
                 fasttext_faiss_index_filename: str,
                 entity_ranker=None,
                 bert_embedder=None,
                 descr_to_emb_filename: str = None,
                 num_ft_faiss_candidate_entities: int = 50,
                 num_tfidf_faiss_candidate_entities: int = 10,
                 num_entities_for_bert_ranking: int = 50,
                 num_tfidf_faiss_cells: int = 50,
                 num_ft_faiss_cells: int = 50,
                 tfidf_index_nprobe: int = 3,
                 fasttext_index_nprobe: int = 3,
                 use_gpu: bool = True,
                 save_path: str = None,
                 fit_tfidf_vectorizer: bool = False,
                 fit_fasttext_vectorizer: bool = False,
                 fit_bert_embedder: bool = False,
                 max_tfidf_features: int = 1000,
                 include_mention: bool = False,
                 ngram_range: List[int] = None,
                 num_entities_to_return: int = 10,
                 lang: str = "ru",
                 use_descriptions: bool = True,
                 return_confidences: bool = False,
                 max_text_len: int = 300,
                 lemmatize: bool = False,
                 full_paragraph: bool = False,
                 max_paragraph_len: int = 100,
                 rank_in_runtime: bool = False,
                 tag_thres_probas: dict = {"PER": 0.79, "LOC": 0.79, "ORG": 0.79},
                 bert_emb_batch_size: int = 100,
                 **kwargs) -> None:
        """

        Args:
            load_path: path to folder with inverted index files
            word_to_idlist_filename: file with dict of words (keys) and entity ids list as value
            entities_ranking_filename: file with dict of entity ids (keys) and number of relations in Wikidata
                for entities
            entities_types_sets_filename: file with entities split into sets of PER, LOC, ORG entity types
            q_to_label_filename: file with labels of entities
            tfidf_vectorizer_filename: filename with TfidfVectorizer data
            tfidf_faiss_index_filename: file with tfidf Faiss index of words
            fasttext_vectorizer_filename: filename with fasttext data
            fasttext_faiss_index_filename: file with fasttext Faiss index of entity titles
            entity_ranker: component deeppavlov.models.kbqa.rel_ranking_bert_infer
            num_faiss_candidate_entities: number of nearest neighbors for the entity substring from the text
            num_entities_for_bert_ranking: number of candidate entities for BERT ranking using description and context
            num_tfidf_faiss_cells: number of Voronoi cells for tfidf Faiss index
            num_ft_faiss_cells: number of Voronoi cells for fasttext Faiss index
            use_gpu: whether to use GPU for faster search of candidate entities
            save_path: path to folder with inverted index files
            fit_tfidf_vectorizer: whether to build tfidf index with Faiss library
            fit_fasttext_vectorizer: whether to build fasttext index with Faiss library
            max_tfidf_features: maximal number of features for TfidfVectorizer
            include_mention: whether to leave entity mention in the context (during BERT ranking)
            ngram_range: char ngrams range for TfidfVectorizer
            num_entities_to_return: number of candidate entities for the substring which are returned
            lang: russian or english
            use_description: whether to perform entity ranking by context and description
            return_confidences: whether to return confidences of entities
            max_text_len: maximum length of text for ranking by context and description
            lemmatize: whether to lemmatize tokens
            **kwargs:
        """
        super().__init__(save_path=save_path, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.word_to_idlist_filename = word_to_idlist_filename
        self.entities_ranking_filename = entities_ranking_filename
        self.entities_types_sets_filename = entities_types_sets_filename
        self.q_to_label_filename = q_to_label_filename
        self.q_to_label_out_filename = q_to_label_out_filename
        self.q_to_descr_filename = q_to_descr_filename
        self.descr_to_emb_filename = descr_to_emb_filename
        self.tfidf_vectorizer_filename = tfidf_vectorizer_filename
        self.tfidf_faiss_index_filename = tfidf_faiss_index_filename
        self.fasttext_vectorizer_filename = fasttext_vectorizer_filename
        self.fasttext_faiss_index_filename = fasttext_faiss_index_filename
        self.num_entities_for_bert_ranking = num_entities_for_bert_ranking
        self.num_tfidf_faiss_candidate_entities = num_tfidf_faiss_candidate_entities
        self.num_ft_faiss_candidate_entities = num_ft_faiss_candidate_entities
        self.num_tfidf_faiss_cells = num_tfidf_faiss_cells
        self.num_ft_faiss_cells = num_ft_faiss_cells
        self.tfidf_index_nprobe = tfidf_index_nprobe
        self.fasttext_index_nprobe = fasttext_index_nprobe
        self.use_gpu = use_gpu
        self.entity_ranker = entity_ranker
        self.bert_embedder = bert_embedder
        self.fit_tfidf_vectorizer = fit_tfidf_vectorizer
        self.fit_fasttext_vectorizer = fit_fasttext_vectorizer
        self.fit_bert_embedder = fit_bert_embedder
        self.max_tfidf_features = max_tfidf_features
        self.include_mention = include_mention
        self.ngram_range = ngram_range
        self.num_entities_to_return = num_entities_to_return
        self.lang_str = f"@{lang}"
        if self.lang_str == "@en":
            self.stopwords = set(stopwords.words("english"))
        elif self.lang_str == "@ru":
            self.stopwords = set(stopwords.words("russian"))
        self.not_found_tokens = ["ооо", "оао", "фгуп", "муп", "акционерное общество", "зао", "мкп"]
        self.not_found_str = "not in wiki"
        self.use_descriptions = use_descriptions
        self.return_confidences = return_confidences
        self.max_text_len = max_text_len
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.full_paragraph = full_paragraph
        self.max_paragraph_len = max_paragraph_len
        self.rank_in_runtime = rank_in_runtime
        self.tag_thres_probas = tag_thres_probas
        self.bert_emb_batch_size = bert_emb_batch_size
        self.q_to_descr = {}
        self.descr_to_emb = {}

        self.load()

        if self.fit_tfidf_vectorizer:
            self.tfidf_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(self.ngram_range),
                                                    max_features=self.max_tfidf_features, max_df=0.85)
            self.tfidf_vectorizer.fit(self.word_list)
            self.matrix = self.tfidf_vectorizer.transform(self.word_list)
            self.dense_matrix = self.matrix.toarray()
            if self.num_tfidf_faiss_cells > 1:
                quantizer = faiss.IndexFlatIP(self.max_tfidf_features)
                self.tfidf_faiss_index = faiss.IndexIVFFlat(quantizer, self.max_tfidf_features,
                                                            self.num_tfidf_faiss_cells)
                self.tfidf_faiss_index.train(self.dense_matrix.astype(np.float32))
            else:
                self.tfidf_faiss_index = faiss.IndexFlatIP(self.max_tfidf_features)
            self.tfidf_faiss_index.add(self.dense_matrix.astype(np.float32))
            self.save_tfidf_vectorizer_data()
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.tfidf_faiss_index = faiss.index_cpu_to_gpu(res, 0, self.tfidf_faiss_index)

        if self.fit_fasttext_vectorizer:
            labels_fasttext_vectors = []
            with tqdm(total=len(self.labels_list)) as pbar:
                for label in self.labels_list:
                    labels_fasttext_vectors.append(self.alies2ft_vec(label))
                    pbar.update(1)
            fasttext_dim = self.fasttext_vectorizer.get_dimension()
            quantizer = faiss.IndexFlatIP(fasttext_dim)
            self.fasttext_faiss_index = faiss.IndexIVFFlat(quantizer, fasttext_dim, self.num_ft_faiss_cells)
            self.fasttext_faiss_index.train(np.array(labels_fasttext_vectors))
            self.fasttext_faiss_index.add(np.array(labels_fasttext_vectors))
            faiss.write_index(self.ft_faiss_index, str(expand_path(self.fasttext_faiss_index_filename)))

        if self.fit_bert_embedder:
            q_to_descr_list = list(self.q_to_descr.items())
            descr_length = len(q_to_descr_list)

            num_chunks = descr_length // self.bert_emb_batch_size + int(descr_length % self.bert_emb_batch_size > 0)
            for chunk_num in range(num_chunks):
                cur_chunk = q_to_descr_list[
                            chunk_num * self.bert_emb_batch_size:(chunk_num + 1) * self.bert_emb_batch_size]
                batch_entities = [el[0] for el in cur_chunk]
                batch_descr = [el[1] for el in cur_chunk]
                context_embs_res = list(self.bert_embedder(batch_descr)[:, :100])
                for cur_entity, emb in zip(batch_entities, context_embs_res):
                    self.descr_to_emb[cur_entity] = emb
            save_pickle(self.descr_to_emb, self.save_path / self.descr_to_emb_filename)

        self.tfidf_faiss_index.nprobe = self.tfidf_index_nprobe
        self.fasttext_faiss_index.nprobe = self.fasttext_index_nprobe

    def load(self) -> None:
        self.word_to_idlist = load_pickle(self.load_path / self.word_to_idlist_filename)
        self.word_list = list(self.word_to_idlist.keys())
        self.entities_ranking_dict = load_pickle(self.load_path / self.entities_ranking_filename)
        self.entities_types_sets = load_pickle(self.load_path / self.entities_types_sets_filename)
        self.q_to_label = load_pickle(self.load_path / self.q_to_label_filename)
        self.q_to_label_out = load_pickle(self.load_path / self.q_to_label_out_filename)
        self.label_to_q = {}
        for q_id in self.q_to_label:
            for label in self.q_to_label[q_id]:
                if label in self.label_to_q:
                    self.label_to_q[label].append(q_id)
                else:
                    self.label_to_q[label] = [q_id]
        self.labels_list = list(self.label_to_q.keys())
        if not self.fit_tfidf_vectorizer:
            self.tfidf_vectorizer = load_pickle(expand_path(self.tfidf_vectorizer_filename))
            self.tfidf_faiss_index = faiss.read_index(str(expand_path(self.tfidf_faiss_index_filename)))
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.tfidf_faiss_index = faiss.index_cpu_to_gpu(res, 0, self.tfidf_faiss_index)

        self.fasttext_vectorizer = fasttext.load_model(str(expand_path(self.fasttext_vectorizer_filename)))
        if not self.fit_fasttext_vectorizer:
            self.fasttext_faiss_index = faiss.read_index(str(expand_path(self.fasttext_faiss_index_filename)))

        if self.q_to_descr_filename:
            self.q_to_descr = load_pickle(self.load_path / self.q_to_descr_filename)

        if self.descr_to_emb_filename:
            self.descr_to_emb = load_pickle(self.load_path / self.descr_to_emb_filename)

    def save(self) -> None:
        pass

    def alies2ft_vec(self, alies):
        if isinstance(alies, str):
            alies = '_'.join(alies.split(' ')).lower()
        elif isinstance(alies, list):
            if alies:
                alies = '_'.join(alies).lower()
            else:
                alies = "_"
        vec = self.fasttext_vectorizer.get_word_vector(alies)
        return vec.astype('float32')

    def save_tfidf_vectorizer_data(self) -> None:
        save_pickle(self.tfidf_vectorizer, expand_path(self.tfidf_vectorizer_filename))
        faiss.write_index(self.tfidf_faiss_index, str(expand_path(self.tfidf_faiss_index_filename)))

    def __call__(self, entity_substr_batch: List[List[str]],
                 entity_offsets_batch: List[List[List[int]]],
                 tags_batch: List[List[str]],
                 sentences_offsets_batch: List[List[Tuple[int, int]]],
                 sentences_batch: List[List[str]],
                 probas_batch: List[List[float]] = None
                 ) -> Tuple[List[List[str]], List[List[List[Tuple[float, int, float]]]],
                            List[List[List[int]]], List[List[List[str]]]]:

        """
        
        Args:
            entity_substr_batch: entity substrings
            entity_offsets_batch: indices of start and end symbols of entities in text
            tags batch: NER tags of entity substrings (PER, LOC, ORG)
            sentences_offsets_batch: indices of start and end symbols of sentences in text
            sentences_batch: list of sentences from texts
            
        Examples of input arguments:
            entity_substr_batch: [['екатеринбург', 'россии', 'уральского федерального округа',
                                   'свердловской области', 'среднего урала', 'исети']]
            entity_offsets_batch: [[(0, 12), (23, 29), (54, 84), (87, 107), (140, 154), (172, 177)]]
            tags_batch: [['LOC', 'LOC', 'LOC', 'LOC', 'LOC', 'LOC']]
            sentences_offsets_batch: [[(0, 108), (109, 178)]]
            sentences_batch: [['Екатеринбург - город в России, административный центр Уральского федерального округа
                                и Свердловской области.',
                               'Расположен на восточном склоне Среднего Урала, по берегам реки Исети.']]
                               
        Examples of outputs:
        If return one candidate entity for each substring (self.num_entities_to_return == 1)
            entity_substr_batch: [['екатеринбург', 'россии', 'уральского федерального округа',
                                   'свердловской области', 'среднего урала', 'исети']]
            conf_batch: [[(1.0, 66, 1.0), (1.0, 203, 1.0), (0.61, 20, 0.91),
                          (1.0, 51, 1.0), (0.5, 9, 1.0), (1.0, 21, 1.0)]]
            entity_offsets_batch: [[(0, 12), (23, 29), (54, 84), (87, 107), (140, 154), (172, 177)]]
            entity_ids_batch: [['Q887', 'Q159', 'Q41964', 'Q5462', 'Q1096949', 'Q874369']]
            
        If return several candidate entities for each substring, for example, two (self.num_entities_to_return == 2)
            entity_substr_batch: [['екатеринбург', 'россии', 'уральского федерального округа',
                                   'свердловской области', 'среднего урала', 'исети']]
            conf_batch: [[[(1.0, 66, 1.0), (0.89, 15, 0.02)], [(1.0, 203, 1.0), (1.0, 12, 0.01)],
                          [(0.61, 20, 0.91), (0.61, 5, 0.16)], [(1.0, 51, 1.0), (0.67, 12, 0.45)],
                          [(0.5, 9, 1.0), (0.5, 12, 0.97)], [(1.0, 21, 1.0), (1.0, 6, 1.0)]]]
            entity_offsets_batch: [[(0, 12), (23, 29), (54, 84), (87, 107), (140, 154), (172, 177)]]
            entity_ids_batch: [[['Q887', 'Q3180012'], ['Q159', 'Q1849069'], ['Q41964', 'Q4476750'],
                                ['Q5462', 'Q1998912'], ['Q1096949', 'Q1894057'], ['Q874369', 'Q36232823']]]
        """
        nf_entity_substr_batch, nf_tags_batch, nf_probas_batch, nf_entity_offsets_batch = [], [], [], []
        nf_entity_ids_batch, nf_conf_batch = [], []
        fnd_entity_substr_batch, fnd_tags_batch, fnd_probas_batch, fnd_entity_offsets_batch = [], [], [], []
        if probas_batch is None:
            probas_batch = [[] for _ in entity_substr_batch]

        for entity_substr_list, tags_list, entity_offsets_list, probas_list in \
                zip(entity_substr_batch, tags_batch, entity_offsets_batch, probas_batch):
            if not probas_list and entity_substr_list:
                probas_list = [0.8 for _ in entity_substr_list]
            nf_entity_substr_list, nf_tags_list, nf_probas_list, nf_entity_offsets_list = [], [], [], []
            nf_entity_ids_list, nf_conf_list = [], []
            fnd_entity_substr_list, fnd_tags_list, fnd_probas_list, fnd_entity_offsets_list = [], [], [], []
            if probas_list is None:
                probas_list = []
            for entity_substr, tag, proba, entity_offsets in zip(entity_substr_list, tags_list, probas_list,
                                                                 entity_offsets_list):
                nf = False
                for tok in self.not_found_tokens:
                    if tok in entity_substr:
                        nf = True
                        break
                if nf:
                    nf_entity_substr_list.append(entity_substr)
                    nf_tags_list.append(tag)
                    nf_probas_list.append(proba)
                    nf_entity_offsets_list.append(entity_offsets)
                    if self.num_entities_to_return == 1:
                        nf_entity_ids_list.append(self.not_found_str)
                        nf_conf_list.append((0.0, 0, 0.0))
                    else:
                        nf_entity_ids_list.append([self.not_found_str])
                        nf_conf_list.append([(0.0, 0, 0.0)])
                else:
                    fnd_entity_substr_list.append(entity_substr)
                    fnd_tags_list.append(tag)
                    fnd_probas_list.append(proba)
                    fnd_entity_offsets_list.append(entity_offsets)
            nf_entity_substr_batch.append(nf_entity_substr_list)
            nf_tags_batch.append(nf_tags_list)
            nf_probas_batch.append(nf_probas_list)
            nf_entity_offsets_batch.append(nf_entity_offsets_list)
            nf_entity_ids_batch.append(nf_entity_ids_list)
            nf_conf_batch.append(nf_conf_list)
            fnd_entity_substr_batch.append(fnd_entity_substr_list)
            fnd_tags_batch.append(fnd_tags_list)
            fnd_probas_batch.append(fnd_probas_list)
            fnd_entity_offsets_batch.append(fnd_entity_offsets_list)

        fnd_entity_ids_batch, fnd_conf_batch = \
            self.link_entities(fnd_entity_substr_batch, fnd_tags_batch, fnd_probas_batch, fnd_entity_offsets_batch,
                               sentences_batch,
                               sentences_offsets_batch)

        entity_substr_batch, tags_batch, entity_offsets_batch, entity_ids_batch, conf_batch = [], [], [], [], []
        for i in range(len(nf_entity_substr_batch)):
            entity_substr_list = nf_entity_substr_batch[i] + fnd_entity_substr_batch[i]
            tags_list = nf_tags_batch[i] + fnd_tags_batch[i]
            entity_offsets_list = nf_entity_offsets_batch[i] + fnd_entity_offsets_batch[i]
            entity_ids_list = nf_entity_ids_batch[i] + fnd_entity_ids_batch[i]
            conf_list = nf_conf_batch[i] + fnd_conf_batch[i]
            entity_substr_batch.append(entity_substr_list)
            tags_batch.append(tags_list)
            entity_offsets_batch.append(entity_offsets_list)
            entity_ids_batch.append(entity_ids_list)
            conf_batch.append(conf_list)
        entity_labels_batch = self.find_labels(entity_ids_batch)
        
        for i in range(len(entity_substr_batch)):
            entity_substr_list = entity_substr_batch[i]
            conf_list = conf_batch[i]
            entity_ids_list = entity_ids_batch[i]
            tags_list = tags_batch[i]
            entity_labels_list = entity_labels_batch[i]
            full_names = {}
            full_names_2 = {}
            for entity_substr, tag, entity_ids, conf, entity_labels in \
                zip(entity_substr_list, tags_list, entity_ids_list, conf_list, entity_labels_list):
                if tag == "PER" and len(entity_substr.split()) > 1:
                    entity_substr_tuple = tuple(entity_substr.split())
                    found_conf = conf[0]
                    if entity_ids != ["not in wiki"]:
                        full_names[entity_substr_tuple] = [tag, entity_ids, conf, entity_labels]
                    if found_conf[1] > 50 and found_conf[2] > 0.3:
                        full_names[entity_substr_tuple] = [tag, entity_ids, conf, entity_labels]
                    if found_conf[1] > 20 and found_conf[2] > 0.2:
                        full_names_2[entity_substr_tuple] = [tag, entity_ids, conf, entity_labels]
                if tag == "PER" and len(entity_substr.split()) == 1:
                    entity_label_split = entity_labels[0].lower().split()
                    entity_substr_tuple = tuple(entity_label_split)
                    found_conf = conf[0]
                    if "," not in entity_labels[0] and "ли" not in entity_label_split \
                            and (entity_substr[:-2].startswith(entity_label_split[-1][:-2]) \
                                 or entity_label_split[-1][:-2].startswith(entity_substr[:-2])) \
                            and len(entity_substr) > 3 and len(entity_label_split[-1]) > 3:
                        if found_conf[1] > 50 and found_conf[2] > 0.3:
                            full_names[entity_substr_tuple] = [tag, entity_ids, conf, entity_labels]
                        if found_conf[1] > 20 and found_conf[2] > 0.2:
                            full_names_2[entity_substr_tuple] = [tag, entity_ids, conf, entity_labels]
                    if "," in entity_labels[0] and "ли" not in entity_label_split \
                            and (entity_substr[:-2].startswith(entity_label_split[0][:-2]) \
                                 or entity_label_split[0][:-2].startswith(entity_substr[:-2])) \
                            and len(entity_substr) > 3 and len(entity_label_split[0]) > 3:
                        if found_conf[1] > 50 and found_conf[2] > 0.3:
                            full_names[entity_substr_tuple] = [tag, entity_ids, conf, entity_labels]
                        if found_conf[1] > 20 and found_conf[2] > 0.2:
                            full_names_2[entity_substr_tuple] = [tag, entity_ids, conf, entity_labels]
                        
            putin = []
            for entity_substr, tag, entity_ids, conf, entity_labels in \
                zip(entity_substr_list, tags_list, entity_ids_list, conf_list, entity_labels_list):
                if "путин" in entity_substr and entity_ids[0] == "Q7747":
                    putin = [tag, entity_ids, conf, entity_labels]
            
            for j in range(len(entity_substr_list)):
                entity_substr = entity_substr_list[j]
                if tags_list[j] == "PER"and len(entity_substr.split()) <= 2:
                    found_full_name = ""
                    entity_substr_split = entity_substr.split()
                    if len(entity_substr.split()) == 1:
                        for full_name in full_names:
                            for elem in full_name:
                                if len(entity_substr) > 3 and len(elem) > 3 \
                                        and (entity_substr[:-2].startswith(elem[:-2]) or elem[:-2].startswith(entity_substr[:-2])):
                                    found_full_name = full_name
                                    break
                            if found_full_name:
                                break
                    
                    if len(entity_substr.split()) == 2 and entity_ids_list[j] == ["not in wiki"]:
                        for full_name in full_names_2:
                            found_cnt = 0
                            for entity_substr_elem in entity_substr_split:
                                for elem in full_name:
                                    if len(entity_substr_elem) > 3 and len(elem) > 3 \
                                            and (entity_substr_elem[:-2].startswith(elem[:-2])
                                                 or elem[:-2].startswith(entity_substr_elem[:-2])):
                                        found_cnt += 1
                                        break
                            if found_cnt == 2:
                                found_full_name = full_name
                                break
                    if found_full_name:
                        found_entity_info = []
                        if found_full_name in full_names:
                            found_entity_info = full_names[found_full_name]
                        if found_full_name in full_names_2:
                            found_entity_info = full_names_2[found_full_name]
                        if found_entity_info:
                            tags_list[j] = found_entity_info[0]
                            entity_ids_list[j] = found_entity_info[1]
                            conf_list[j] = found_entity_info[2]
                            entity_labels_list[j] = found_entity_info[3]
                
                if entity_substr.lower() == "ввп" and putin:
                    tags_list[j] = putin[0]
                    entity_ids_list[j] = putin[1]
                    conf_list[j] = putin[2]
                    entity_labels_list[j] = putin[3]
            
            conf_batch[i] = conf_list
            entity_ids_batch[i] = entity_ids_list
            tags_batch[i] = tags_list
            entity_labels_batch[i] = entity_labels_list
            
        status_batch = []
        for entity_ids_list in entity_ids_batch:
            if entity_ids_list and entity_ids_list[0] \
                    and (entity_ids_list[0][0] == "ERROR" or entity_ids_list[0][0] == ["ERROR"]):
                status_batch.append("error")
            else:
                status_batch.append("ok")

        if self.return_confidences:
            return entity_substr_batch, conf_batch, entity_offsets_batch, entity_ids_batch, tags_batch, \
                   entity_labels_batch, status_batch
        else:
            return entity_substr_batch, entity_offsets_batch, entity_ids_batch, tags_batch, entity_labels_batch, status_batch

    def link_entities(self, entity_substr_batch: List[List[str]],
                      tags_batch: List[List[str]],
                      probas_batch: List[List[float]],
                      entity_offsets_batch: List[List[List[int]]],
                      sentences_batch: List[List[str]],
                      sentences_offsets_batch: List[List[Tuple[int, int]]]) -> List[List[List[Tuple[int, int]]]]:
        log.debug(f"entity substr batch {entity_substr_batch}")
        log.debug(f"entity offsets batch {entity_offsets_batch}")
        entity_substr_batch = [[entity_substr.replace('"', '') for entity_substr in entity_substr_list] for entity_substr_list in entity_substr_batch]
        entity_substr_batch = [[[word for word in entity_substr.split(' ')
                                 if (word not in self.stopwords or word == "ли") and len(word) > 0 and word not in punctuation]
                                for entity_substr in entity_substr_list]
                               for entity_substr_list in entity_substr_batch]

        entity_ids_batch = []
        conf_batch = []
        for entity_substr_list, entity_offsets_list, sentences_list, sentences_offsets_list, tags_list, probas_list in \
                zip(entity_substr_batch, entity_offsets_batch, sentences_batch, sentences_offsets_batch, tags_batch,
                    probas_batch):
            entity_ids_list, conf_list = [], []
            if entity_substr_list:
                try:
                    tm_ind_st = time.time()
                    ft_entity_emb_list = [self.alies2ft_vec(entity_substr) for entity_substr in entity_substr_list]
                    ft_res = []
                    if ft_entity_emb_list:
                        ft_res = self.fasttext_faiss_index.search(np.array(ft_entity_emb_list),
                                                                  self.num_ft_faiss_candidate_entities)
                    D_ft_all, I_ft_all = [], []
                    if len(ft_res) == 2:
                        D_ft_all, I_ft_all = ft_res
                    words = []
                    entity_substr_num = []
                    for i, entity_substr in enumerate(entity_substr_list):
                        words += entity_substr
                        entity_substr_num += [i for _ in entity_substr]

                    ent_substr_tfidfs = self.tfidf_vectorizer.transform(words).toarray().astype(np.float32)
                    D_all, I_all = self.tfidf_faiss_index.search(ent_substr_tfidfs, self.num_tfidf_faiss_candidate_entities)

                    ind_i = 0
                    candidate_entities_dict = {index: [] for index in range(len(entity_substr_list))}
                    candidate_entities_ft_total = [[] for _ in entity_substr_list]
                    substr_lens = [len(entity_substr) for entity_substr in entity_substr_list]
                    for i, (entity_substr, tag, proba, cand_entity_len) in \
                            enumerate(zip(entity_substr_list, tags_list, probas_list, substr_lens)):
                        for word in entity_substr:
                            candidate_entities = {}
                            entities_set = set()
                            morph_parsed_word = self.morph_parse(word)
                            if word in self.word_to_idlist or morph_parsed_word in self.word_to_idlist:
                                entities_set = self.word_to_idlist.get(word, set())
                                if tag == "ORG":
                                    entities_set_1 = self.filter_entities_by_tags(entities_set, "ORG", proba)
                                    entities_set_2 = self.filter_entities_by_tags(entities_set, "LOC", proba)
                                    entities_set = entities_set_1.union(entities_set_2)
                                else:
                                    entities_set = self.filter_entities_by_tags(entities_set, tag, proba)
                                if word != morph_parsed_word:
                                    entities_set = entities_set.union(self.word_to_idlist.get(morph_parsed_word, set()))
                                    if tag == "ORG":
                                        entities_set_1 = self.filter_entities_by_tags(entities_set, "ORG", proba)
                                        entities_set_2 = self.filter_entities_by_tags(entities_set, "LOC", proba)
                                        entities_set = entities_set_1.union(entities_set_2)
                                    else:
                                        entities_set = self.filter_entities_by_tags(entities_set, tag, proba)
                                for entity in entities_set:
                                    candidate_entities[entity] = 1.0
                            else:
                                scores_list = D_all[ind_i]
                                ind_list = I_all[ind_i]
                                if self.num_tfidf_faiss_cells > 1:
                                    scores_list = [1.0 - score for score in scores_list]
                                for ind, score in zip(ind_list, scores_list):
                                    entities_set = self.word_to_idlist[self.word_list[ind]]
                                    entities_set = self.filter_entities_by_tags(entities_set, tag, proba)
                                    for entity in entities_set:
                                        if entity in candidate_entities:
                                            if score > candidate_entities[entity]:
                                                candidate_entities[entity] = score
                                        else:
                                            candidate_entities[entity] = score
                            candidate_entities_dict[i] += [(entity, cand_entity_len, score)
                                                           for (entity, cand_entity_len), score
                                                           in candidate_entities.items()]
                            ind_i += 1

                    if isinstance(D_ft_all, np.ndarray):
                        candidate_entities_ft_dict = {index: [] for index in range(len(entity_substr_list))}
                        for index, (entity_substr, scores_list, ind_list, tag, proba) \
                                in enumerate(zip(entity_substr_list, D_ft_all, I_ft_all, tags_list, probas_list)):
                            entities_set = set()
                            for ind, score in zip(ind_list, scores_list):
                                if score < 400.0:
                                    entity_label = self.labels_list[ind]
                                    cur_entity_tokens = set([token.lower() for token in entity_substr])
                                    cur_entity_label_tokens = set(entity_label.lower().split())
                                    inters_tokens = cur_entity_tokens.intersection(cur_entity_label_tokens)
                                    num_matches = 0
                                    for cur_entity_tok in cur_entity_tokens:
                                        for cur_entity_label_tok in cur_entity_label_tokens:
                                            if cur_entity_tok[:2] == cur_entity_label_tok[:2] \
                                                    and (fuzz.ratio(cur_entity_tok, cur_entity_label_tok) > 70.0 \
                                                    or (len(cur_entity_tok) == 3 and len(cur_entity_label_tok) == 3)):
                                                num_matches += 1
                                                break
                                    if (len(cur_entity_tokens) == 1 and num_matches == 1) \
                                            or (len(cur_entity_tokens) == 2 and num_matches == 2) \
                                            or (len(cur_entity_tokens) > 2 and abs(num_matches - len(cur_entity_tokens)) <= 1):
                                        if inters_tokens:
                                            fuzz_ratio = len(inters_tokens) / max(len(cur_entity_tokens),
                                                                                  len(cur_entity_label_tokens))
                                        else:
                                            fuzz_ratio = fuzz.ratio(' '.join(entity_substr).lower(),
                                                                    entity_label.lower()) * 0.01
                                        for entity_id in self.label_to_q[entity_label]:
                                            entities_set.add((entity_id, fuzz_ratio))
                            entities_set = self.filter_entities_by_tags(entities_set, tag, proba)
                            candidate_entities_ft_dict[index] = list(entities_set)
                        candidate_entities_ft_total = list(candidate_entities_ft_dict.values())

                    candidate_entities_total = candidate_entities_dict.values()
                    candidate_entities_total = [self.sum_scores(candidate_entities, substr_len)
                                                for candidate_entities, substr_len in
                                                zip(candidate_entities_total, substr_lens)]
                    candidate_entities_total = [list(candidate_entities) for candidate_entities in candidate_entities_total]

                    log.debug(f"length candidate entities list {len(candidate_entities_total)}")
                    
                    candidate_entities_list = []
                    entities_scores_list = []
                    for entity_substr, candidate_entities, candidate_entities_ft \
                            in zip(entity_substr_list, candidate_entities_total, candidate_entities_ft_total):
                        log.debug(f"{entity_substr} candidate_entities before ranking {candidate_entities[:10]}")
                        candidate_entities_dict = {}
                        for entity, score in candidate_entities:
                            candidate_entities_dict[entity] = score
                        for entity, fuzz_score in candidate_entities_ft:
                            if entity in candidate_entities_dict:
                                score = candidate_entities_dict[entity]
                                candidate_entities_dict[entity] = max(score, fuzz_score)
                            else:
                                candidate_entities_dict[entity] = fuzz_score
                        candidate_entities = candidate_entities_dict.items()

                        candidate_entities = [candidate_entity + (self.entities_ranking_dict.get(candidate_entity[0], 0),)
                                              for candidate_entity in candidate_entities]
                        candidate_entities = sorted(candidate_entities, key=lambda x: (x[1], x[2]), reverse=True)

                        log.debug(f"candidate_entities {candidate_entities[:10]}")
                        entities_scores = {entity: (substr_score, pop_score)
                                           for entity, substr_score, pop_score in candidate_entities}
                        candidate_entities = [candidate_entity[0] for candidate_entity
                                              in candidate_entities][:self.num_entities_for_bert_ranking]
                        conf = [candidate_entity[1:] for candidate_entity
                                in candidate_entities][:self.num_entities_for_bert_ranking]
                        log.debug(f"{entity_substr} candidate_entities before bert ranking {candidate_entities[:10]}")
                        candidate_entities_list.append(candidate_entities)
                        if self.num_entities_to_return == 1 and candidate_entities:
                            entity_ids_list.append(candidate_entities[0])
                            conf_list.append(conf[0])
                        else:
                            entity_ids_list.append(candidate_entities[:self.num_entities_to_return])
                            conf_list.append(conf[:self.num_entities_to_return])
                        entities_scores_list.append(entities_scores)
                    tm_ind_end = time.time()
                    log.debug(f"search by index time {tm_ind_end - tm_ind_st}")
                    tm_descr_st = time.time()
                    if self.use_descriptions:
                        if self.rank_in_runtime:
                            entity_ids_list, conf_list = self.rank_by_description_runtime(entity_substr_list,
                                                                                          entity_offsets_list,
                                                                                          candidate_entities_list,
                                                                                          tags_list,
                                                                                          entities_scores_list,
                                                                                          sentences_list,
                                                                                          sentences_offsets_list,
                                                                                          substr_lens)
                        else:
                            entity_ids_list, conf_list = self.rank_by_description(entity_substr_list, entity_offsets_list,
                                                                                  candidate_entities_list, tags_list,
                                                                                  entities_scores_list, sentences_list,
                                                                                  sentences_offsets_list, substr_lens)
                    tm_descr_end = time.time()
                    log.debug(f"description time {tm_descr_end - tm_descr_st}")
                except:
                    if self.num_entities_to_return == 1:
                        entity_ids_list = ["ERROR" for _ in entity_substr_list]
                        conf_list = [(0.0, 0, 0.0) for _ in entity_substr_list]
                    else:
                        entity_ids_list = [["ERROR"] for _ in entity_substr_list]
                        conf_list = [[(0.0, 0, 0.0)] for _ in entity_substr_list]
                if entity_substr_list and entity_ids_list[0] == []:
                    entity_ids_list = [["Not Found"] for _ in entity_substr_list]
                    conf_list = [[(0.0, 0, 0.0)] for _ in entity_substr_list]
                
                corr_entity_ids_list = []
                corr_conf_list = []
                for entity_ids, conf in zip(entity_ids_list, conf_list):
                    if entity_ids == []:
                        corr_entity_ids_list.append(["Not Found"])
                        corr_conf_list.append([(0.0, 0, 0.0)])
                    else:
                        corr_entity_ids_list.append(entity_ids)
                        corr_conf_list.append(conf)
                entity_ids_list = corr_entity_ids_list
                conf_list = corr_conf_list

            entity_ids_batch.append(entity_ids_list)
            conf_batch.append(conf_list)

        return entity_ids_batch, conf_batch

    def morph_parse(self, word):
        morph_parse_tok = self.morph.parse(word)[0]
        if morph_parse_tok.tag.POS in {"NOUN", "ADJ", "ADJF"}:
            normal_form = morph_parse_tok.inflect({"nomn"}).word
        else:
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
        entities_with_scores = set(entities_with_scores.items())

        return entities_with_scores

    def rank_by_description(self, entity_substr_list: List[str],
                            entity_offsets_list: List[List[int]],
                            candidate_entities_list: List[List[str]],
                            tags: List[str],
                            entities_scores_list: List[Dict[str, Tuple[int, float]]],
                            sentences_list: List[str],
                            sentences_offsets_list: List[Tuple[int, int]],
                            substr_lens: List[int]) -> List[List[str]]:
        log.debug(f"rank, entity pos {entity_offsets_list}")
        log.debug(f"rank, sentences_list {sentences_list}")
        log.debug(f"rank, sent offsets {sentences_offsets_list}")
        log.debug(f"rank, substr_lens {substr_lens}")
        entity_ids_list = []
        conf_list = []
        contexts = []
        sentences_to_nums = {}
        samples_to_contexts = []
        contexts_count = 0
        for sample_num, (entity_substr, (entity_start_offset, entity_end_offset), candidate_entities) in \
                enumerate(zip(entity_substr_list, entity_offsets_list, candidate_entities_list)):
            context_sent_nums = set()
            log.debug(f"entity_offsets {entity_start_offset}, {entity_end_offset}")
            log.debug(f"candidate_entities {candidate_entities[:10]}")
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
            log.debug(f"rank, found sentence {sentence}")
            log.debug(f"rank, relative offsets {rel_start_offset}, {rel_end_offset}")
            context = ""
            if sentence:
                context_sent_nums.add(found_sentence_num)
                start_of_sentence = 0
                end_of_sentence = len(sentence)
                if len(sentence) > self.max_text_len:
                    start_of_sentence = max(rel_start_offset - self.max_text_len // 2, 0)
                    end_of_sentence = min(rel_end_offset + self.max_text_len // 2, len(sentence))
                if self.include_mention:
                    context = sentence[start_of_sentence:rel_start_offset] + "[ENT]" + \
                              sentence[rel_start_offset:rel_end_offset] + "[ENT]" + \
                              sentence[rel_end_offset:end_of_sentence]
                else:
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
                                context_sent_nums.add(last_sentence_num + 1)
                                last_sentence_num += 1
                                added = True
                        if first_sentence_num > 0:
                            first_sentence_len = len(
                                re.findall(self.re_tokenizer, sentences_list[first_sentence_num - 1]))
                            if cur_sent_len + first_sentence_len < self.max_paragraph_len:
                                context = [sentences_list[first_sentence_num - 1]] + context
                                cur_sent_len += first_sentence_len
                                context_sent_nums.add(first_sentence_num - 1)
                                first_sentence_num -= 1
                                added = True
                        if not added:
                            break
                    context = ' '.join(context)

            log.debug(f"rank, context: {context}")
            context_sent_nums = str(context_sent_nums)
            if context_sent_nums not in sentences_to_nums:
                sentences_to_nums[context_sent_nums] = contexts_count
                samples_to_contexts.append(contexts_count)
                contexts_count += 1
                contexts.append(context)
            else:
                samples_to_contexts.append(sentences_to_nums[context_sent_nums])

        scores_list = []

        context_embs_res = self.bert_embedder(contexts)[:, :100]
        context_embs = []
        for num in samples_to_contexts:
            context_embs.append(context_embs_res[num])

        for entity_substr, candidate_entities, context_emb in \
                zip(entity_substr_list, candidate_entities_list, context_embs):
            candidate_entities_emb = [self.descr_to_emb.get(entity, np.zeros(100, dtype=float)) for entity in
                                      candidate_entities]
            scores = [np.dot(candidate_entity_emb, context_emb) for candidate_entity_emb in candidate_entities_emb]
            scores = [max(min((score + 13.0) * 0.05, 1.0), 0.0) for score in scores]
            scores = [(entity, round(score, 4)) for entity, score in zip(candidate_entities, scores)]
            scores_list.append(scores)

        for entity_substr, candidate_entities, tag, substr_len, entities_scores, scores in \
                zip(entity_substr_list, candidate_entities_list, tags, substr_lens, entities_scores_list, scores_list):
            log.debug(f"len candidate entities {len(candidate_entities)}")
            entities_with_scores = [(entity, round(entities_scores.get(entity, (0.0, 0))[0], 2),
                                     entities_scores.get(entity, (0.0, 0))[1],
                                     round(score, 4)) for entity, score in scores]
            log.debug(f"len entities with scores {len(entities_with_scores)}")
            entities_with_scores = [entity for entity in entities_with_scores
                                    if ((entity[3] > 0.1 or entity[1] == 1.0 or entity[2] > 60)
                                        and entity[0].startswith("Q"))]
            entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            log.debug(f"entities_with_scores {entities_with_scores}")

            if not entities_with_scores:
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            elif entities_with_scores and substr_len == 1 and entities_with_scores[0][1] < 1.0:
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            elif entities_with_scores and ((entities_with_scores[0][3] < 0.1 and (
                    entities_with_scores[0][2] < 90 and entities_with_scores[0][1] < 1.0))
                                           or entities_with_scores[0][1] < 0.3
                                           or (entities_with_scores[0][3] < 0.019 and entities_with_scores[0][2] < 20)
                                           or (entities_with_scores[0][3] < 0.1 and entities_with_scores[0][2] < 4)
                                           or (entities_with_scores[0][1] == 0.5 and tag in {"PER", "LOC"})):
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            else:
                top_entities = [score[0] for score in entities_with_scores]
                top_conf = [score[1:] for score in entities_with_scores]

            high_conf_entities = []
            high_conf_nums = []
            for elem_num, (entity, conf) in enumerate(zip(top_entities, top_conf)):
                if len(conf) == 3 and conf[0] == 1.0 and conf[1] > 29 and conf[2] > 0.019:
                    new_conf = list(conf)
                    if new_conf[1] > 55:
                        new_conf[2] = 1.0
                    else:
                        new_conf[2] = 0.98
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

    def rank_by_description_runtime(self, entity_substr_list: List[str],
                                    entity_offsets_list: List[List[int]],
                                    candidate_entities_list: List[List[str]],
                                    tags: List[str],
                                    entities_scores_list: List[Dict[str, Tuple[int, float]]],
                                    sentences_list: List[str],
                                    sentences_offsets_list: List[Tuple[int, int]],
                                    substr_lens: List[int]) -> List[List[str]]:
        log.debug(f"rank, entity pos {entity_offsets_list}")
        log.debug(f"rank, sentences_list {sentences_list}")
        log.debug(f"rank, sent offsets {sentences_offsets_list}")
        log.debug(f"rank, substr_lens {substr_lens}")
        entity_ids_list = []
        conf_list = []
        contexts = []
        for entity_substr, (entity_start_offset, entity_end_offset), candidate_entities in \
                zip(entity_substr_list, entity_offsets_list, candidate_entities_list):
            log.debug(f"entity_offsets {entity_start_offset}, {entity_end_offset}")
            log.debug(f"candidate_entities {candidate_entities[:10]}")
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
            log.debug(f"rank, found sentence {sentence}")
            log.debug(f"rank, relative offsets {rel_start_offset}, {rel_end_offset}")
            context = ""
            if sentence:
                start_of_sentence = 0
                end_of_sentence = len(sentence)
                if len(sentence) > self.max_text_len:
                    start_of_sentence = max(rel_start_offset - self.max_text_len // 2, 0)
                    end_of_sentence = min(rel_end_offset + self.max_text_len // 2, len(sentence))
                if self.include_mention:
                    context = sentence[start_of_sentence:rel_start_offset] + "[ENT]" + \
                              sentence[rel_start_offset:rel_end_offset] + "[ENT]" + \
                              sentence[rel_end_offset:end_of_sentence]
                else:
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

            log.debug(f"rank, context: {context}")
            contexts.append(context)

        if hasattr(self.entity_ranker, "batch_rank_rels"):
            scores_list = self.entity_ranker.batch_rank_rels(contexts, candidate_entities_list)
        else:
            scores_list = self.entity_ranker(contexts, candidate_entities_list)

        for entity_substr, candidate_entities, tag, substr_len, entities_scores, scores in \
                zip(entity_substr_list, candidate_entities_list, tags, substr_lens, entities_scores_list, scores_list):
            log.debug(f"len candidate entities {len(candidate_entities)}")
            entities_with_scores = [(entity, round(entities_scores.get(entity, (0.0, 0))[0], 2),
                                     entities_scores.get(entity, (0.0, 0))[1],
                                     round(score, 2)) for entity, score in scores]
            log.debug(f"len entities with scores {len(entities_with_scores)}")
            entities_with_scores = [entity for entity in entities_with_scores
                                    if (entity[3] > 0.1 and entity[1] > 0.6 and entity[0].startswith("Q"))]
            entities_with_scores = sorted(entities_with_scores, key=lambda x: (x[1], x[3], x[2]), reverse=True)
            log.debug(f"entities_with_scores {entities_with_scores}")

            if not entities_with_scores:
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            elif entities_with_scores and substr_len == 1 and entities_with_scores[0][1] < 1.0:
                top_entities = [self.not_found_str]
                top_conf = [(0.0, 0, 0.0)]
            elif entities_with_scores and ((entities_with_scores[0][3] < 0.11 and entities_with_scores[0][2] < 90)
                                           or entities_with_scores[0][1] < 0.3
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
        
    def first_element(self, entity_labels):
        if isinstance(entity_labels, list):
            return entity_labels[0]
        return entity_labels

    def find_labels(self, entity_ids_batch: List[List[List[str]]]):
        entity_labels_batch = []
        for entity_ids_list in entity_ids_batch:
            entity_labels_list = []
            for entity_ids in entity_ids_list:
                if isinstance(entity_ids, list):
                    entity_labels = [self.first_element(self.q_to_label_out.get(entity_id, [entity_id])) for entity_id in entity_ids]
                elif isinstance(entity_ids, str):
                    entity_labels = self.first_element(self.q_to_label_out.get(entity_ids, entity_ids))
                else:
                    entity_labels = ["not in wiki" for _ in entity_ids]
                entity_labels_list.append(entity_labels)
            entity_labels_batch.append(entity_labels_list)
        return entity_labels_batch

    def filter_entities_by_tags(self, entities_set, tag, proba):
        if proba > self.tag_thres_probas[tag]:
            entities_set = {entity for entity in entities_set if (entity[0] in self.entities_types_sets[tag]
                                                                  or entity[0] in self.entities_types_sets["AMB"])}
        return entities_set
