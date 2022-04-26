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
from logging import getLogger
from string import punctuation
from typing import List, Tuple

from nltk import sent_tokenize
from transformers import AutoTokenizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.models.entity_extraction.entity_detection_parser import EntityDetectionParser

log = getLogger(__name__)


@register('ner_chunker')
class NerChunker(Component):
    """
        Class to split documents into chunks of max_chunk_len symbols so that the length will not exceed
        maximal sequence length to feed into BERT
    """

    def __init__(self, vocab_file: str, max_seq_len: int = 400, lowercase: bool = False, max_chunk_len: int = 180,
                 batch_size: int = 2, **kwargs):
        """
        Args:
            max_chunk_len: maximal length of chunks into which the document is split
            batch_size: how many chunks are in batch
        """
        self.max_seq_len = max_seq_len
        self.max_chunk_len = max_chunk_len
        self.batch_size = batch_size
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file,
                                                       do_lower_case=True)
        self.punct_ext = punctuation + " " + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.russian_letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.lowercase = lowercase

    def __call__(self, docs_batch: List[str]) -> Tuple[List[List[str]], List[List[int]],
                                                       List[List[List[Tuple[int, int]]]], List[List[List[str]]]]:
        """
        This method splits each document in the batch into chunks wuth the maximal length of max_chunk_len
 
        Args:
            docs_batch: batch of documents
        Returns:
            batch of lists of document chunks for each document
            batch of lists of numbers of documents which correspond to chunks
        """
        text_batch_list, nums_batch_list, sentences_offsets_batch_list, sentences_batch_list = [], [], [], []
        text_batch, nums_batch, sentences_offsets_batch, sentences_batch = [], [], [], []
        for n, doc in enumerate(docs_batch):
            if self.lowercase:
                doc = doc.lower()
            start = 0
            text = ""
            sentences_list = []
            sentences_offsets_list = []
            cur_len = 0
            doc_pieces = doc.split("\n")
            doc_pieces = [self.sanitize(doc_piece) for doc_piece in doc_pieces]
            doc_pieces = [doc_piece for doc_piece in doc_pieces if len(doc_piece) > 1]
            if doc_pieces:
                sentences = []
                for doc_piece in doc_pieces:
                    sentences += sent_tokenize(doc_piece)
                for sentence in sentences:
                    sentence_tokens = re.findall(self.re_tokenizer, sentence)
                    sentence_len = sum([len(self.tokenizer.encode_plus(token, add_special_tokens=False)["input_ids"])
                                        for token in sentence_tokens])
                    if cur_len + sentence_len < self.max_seq_len:
                        text += f"{sentence} "
                        cur_len += sentence_len
                        end = start + len(sentence)
                        sentences_offsets_list.append((start, end))
                        sentences_list.append(sentence)
                        start = end + 1
                    else:
                        text = text.strip()
                        if text:
                            text_batch.append(text)
                            sentences_offsets_batch.append(sentences_offsets_list)
                            sentences_batch.append(sentences_list)
                            nums_batch.append(n)

                        if sentence_len < self.max_seq_len:
                            text = f"{sentence} "
                            cur_len = sentence_len
                            start = 0
                            end = start + len(sentence)
                            sentences_offsets_list = [(start, end)]
                            sentences_list = [sentence]
                            start = end + 1
                        else:
                            text = ""
                            sentence_chunks = sentence.split(" ")
                            for chunk in sentence_chunks:
                                chunk_tokens = re.findall(self.re_tokenizer, chunk)
                                chunk_len = sum([len(self.tokenizer.encode_plus(token,
                                                                                add_special_tokens=False)["input_ids"])
                                                 for token in chunk_tokens])
                                if cur_len + chunk_len < self.max_seq_len:
                                    text += f"{chunk} "
                                    cur_len += chunk_len + 1
                                    end = start + len(chunk)
                                    sentences_offsets_list.append((start, end))
                                    sentences_list.append(chunk)
                                    start = end + 1
                                else:
                                    text = text.strip()
                                    if text:
                                        text_batch.append(text)
                                        sentences_offsets_batch.append(sentences_offsets_list)
                                        sentences_batch.append(sentences_list)
                                        nums_batch.append(n)

                                    text = f"{chunk} "
                                    cur_len = chunk_len
                                    start = 0
                                    end = start + len(chunk)
                                    sentences_offsets_list = [(start, end)]
                                    sentences_list = [chunk]
                                    start = end + 1

                text = text.strip().strip(",")
                if text:
                    text_batch.append(text)
                    nums_batch.append(n)
                    sentences_offsets_batch.append(sentences_offsets_list)
                    sentences_batch.append(sentences_list)
            else:
                text_batch.append("а")
                nums_batch.append(n)
                sentences_offsets_batch.append([(0, len(doc))])
                sentences_batch.append([doc])

        num_batches = len(text_batch) // self.batch_size + int(len(text_batch) % self.batch_size > 0)
        for jj in range(num_batches):
            text_batch_list.append(text_batch[jj * self.batch_size:(jj + 1) * self.batch_size])
            nums_batch_list.append(nums_batch[jj * self.batch_size:(jj + 1) * self.batch_size])
            sentences_offsets_batch_list.append(
                sentences_offsets_batch[jj * self.batch_size:(jj + 1) * self.batch_size])
            sentences_batch_list.append(sentences_batch[jj * self.batch_size:(jj + 1) * self.batch_size])

        return text_batch_list, nums_batch_list, sentences_offsets_batch_list, sentences_batch_list

    def sanitize(self, text):
        text_len = len(text)

        if text_len > 0 and text[text_len - 1] not in {'.', '!', '?'}:
            i = text_len - 1
            while text[i] in self.punct_ext and i > 0:
                i -= 1
                if (text[i] in {'.', '!', '?'} and text[i - 1].lower() in self.russian_letters) or \
                        (i > 1 and text[i] in {'.', '!', '?'} and text[i - 1] in '"' and text[
                            i - 2].lower() in self.russian_letters):
                    break

            text = text[:i + 1]
        text = re.sub(r'\s+', ' ', text)
        return text


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
            ner_parser: component deeppavlov.models.entity_extraction.entity_detection_parser
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
        """
        entity_substr_batch_list, entity_offsets_batch_list, entity_positions_batch_list, tags_batch_list, \
        entity_probas_batch_list, text_len_batch_list, text_tokens_len_batch_list = [], [], [], [], [], [], []
        for text_batch, sentences_offsets_batch, sentences_batch in \
                zip(text_batch_list, sentences_offsets_batch_list, sentences_batch_list):
            text_batch = [text.replace("\xad", " ") for text in text_batch]

            ner_tokens_batch, ner_tokens_offsets_batch, ner_probas_batch, probas_batch = self.ner(text_batch)
            entity_substr_batch, entity_positions_batch, entity_probas_batch = \
                self.ner_parser(ner_tokens_batch, ner_probas_batch, probas_batch)

            entity_pos_tags_probas_batch = [[(entity_substr.lower(), entity_substr_positions, tag, entity_proba)
                                             for tag, entity_substr_list in entity_substr_dict.items()
                                             for entity_substr, entity_substr_positions, entity_proba in
                                             zip(entity_substr_list, entity_positions_dict[tag],
                                                 entity_probas_dict[tag])]
                                            for entity_substr_dict, entity_positions_dict, entity_probas_dict in
                                            zip(entity_substr_batch, entity_positions_batch, entity_probas_batch)]

            entity_substr_batch, entity_offsets_batch, entity_positions_batch, tags_batch, \
            probas_batch = [], [], [], [], []
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
                    entity_substr_list, entity_offsets_list, entity_positions_list = [], [], []
                    tags_list, probas_list = [], []
                entity_substr_batch.append(list(entity_substr_list))
                entity_offsets_batch.append(list(entity_offsets_list))
                entity_positions_batch.append(list(entity_positions_list))
                tags_batch.append(list(tags_list))
                probas_batch.append(list(probas_list))

            entity_substr_batch_list.append(entity_substr_batch)
            tags_batch_list.append(tags_batch)
            entity_offsets_batch_list.append(entity_offsets_batch)
            entity_positions_batch_list.append(entity_positions_batch)
            entity_probas_batch_list.append(probas_batch)
            text_len_batch_list.append([len(text) for text in text_batch])
            text_tokens_len_batch_list.append([len(ner_tokens) for ner_tokens in ner_tokens_batch])

        doc_entity_substr_batch, doc_tags_batch, doc_entity_offsets_batch, doc_probas_batch = [], [], [], []
        doc_entity_positions_batch, doc_sentences_offsets_batch, doc_sentences_batch = [], [], []
        doc_entity_substr, doc_tags, doc_probas, doc_entity_offsets, doc_entity_positions = [], [], [], [], []
        doc_sentences_offsets, doc_sentences = [], []
        cur_doc_num = 0
        text_len_sum = 0
        text_tokens_len_sum = 0
        for entity_substr_batch, tags_batch, probas_batch, entity_offsets_batch, entity_positions_batch, \
            sentences_offsets_batch, sentences_batch, text_len_batch, text_tokens_len_batch, nums_batch in \
                zip(entity_substr_batch_list, tags_batch_list, entity_probas_batch_list, entity_offsets_batch_list,
                    entity_positions_batch_list, sentences_offsets_batch_list, sentences_batch_list,
                    text_len_batch_list, text_tokens_len_batch_list, nums_batch_list):
            for entity_substr_list, tag_list, probas_list, entity_offsets_list, entity_positions_list, \
                sentences_offsets_list, sentences_list, text_len, text_tokens_len, doc_num in \
                    zip(entity_substr_batch, tags_batch, probas_batch, entity_offsets_batch, entity_positions_batch,
                        sentences_offsets_batch, sentences_batch, text_len_batch, text_tokens_len_batch, nums_batch):
                if doc_num == cur_doc_num:
                    doc_entity_substr += entity_substr_list
                    doc_tags += tag_list
                    doc_probas += probas_list
                    doc_entity_offsets += [(start_offset + text_len_sum, end_offset + text_len_sum)
                                           for start_offset, end_offset in entity_offsets_list]
                    doc_sentences_offsets += [(start_offset + text_len_sum, end_offset + text_len_sum)
                                              for start_offset, end_offset in sentences_offsets_list]
                    doc_entity_positions += [[pos + text_tokens_len_sum for pos in positions]
                                             for positions in entity_positions_list]
                    doc_sentences += sentences_list
                    text_len_sum += text_len + 1
                    text_tokens_len_sum += text_tokens_len
                else:
                    doc_entity_substr_batch.append(doc_entity_substr)
                    doc_tags_batch.append(doc_tags)
                    doc_probas_batch.append(doc_probas)
                    doc_entity_offsets_batch.append(doc_entity_offsets)
                    doc_entity_positions_batch.append(doc_entity_positions)
                    doc_sentences_offsets_batch.append(doc_sentences_offsets)
                    doc_sentences_batch.append(doc_sentences)
                    doc_entity_substr = entity_substr_list
                    doc_tags = tag_list
                    doc_probas = probas_list
                    doc_entity_offsets = entity_offsets_list
                    doc_sentences_offsets = sentences_offsets_list
                    doc_sentences = sentences_list
                    cur_doc_num = doc_num
                    text_len_sum = text_len + 1
                    text_tokens_len_sum = text_tokens_len

        doc_entity_substr_batch.append(doc_entity_substr)
        doc_tags_batch.append(doc_tags)
        doc_probas_batch.append(doc_probas)
        doc_entity_offsets_batch.append(doc_entity_offsets)
        doc_entity_positions_batch.append(doc_entity_positions)
        doc_sentences_offsets_batch.append(doc_sentences_offsets)
        doc_sentences_batch.append(doc_sentences)

        return doc_entity_substr_batch, doc_entity_offsets_batch, doc_entity_positions_batch, doc_tags_batch, \
               doc_sentences_offsets_batch, doc_sentences_batch, doc_probas_batch
