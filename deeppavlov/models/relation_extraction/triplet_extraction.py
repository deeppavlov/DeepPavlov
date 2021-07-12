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

import itertools
import re
import random
import time
from logging import getLogger
from typing import Tuple, List, Optional, Union, Dict, Any

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
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
        entity_substr_batch_list = []
        entity_offsets_batch_list = []
        tags_batch_list = []
        entity_probas_batch_list = []
        text_len_batch_list = []
        ner_tokens_batch_list = []
        entity_positions_batch_list = []
        sentences_tokens_batch_list = []
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
            pr_entity_positions_batch = []
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
                    entity_substr_list, entity_offsets_list, tags_list, probas_list, entity_positions_list = [], [], [], [], []
                entity_substr_batch.append(list(entity_substr_list))
                entity_offsets_batch.append(list(entity_offsets_list))
                tags_batch.append(list(tags_list))
                probas_batch.append(list(probas_list))
                pr_entity_positions_batch.append(list(entity_positions_list))

            sentences_tokens_batch = []
            for sentences_offsets_list, ner_tokens_list, ner_tokens_offsets_list in \
                    zip(sentences_offsets_batch, ner_tokens_batch, ner_tokens_offsets_batch):
                sentences_tokens_list = []
                for start_offset, end_offset in sentences_offsets_list:
                    sentence_tokens = []
                    for tok, (start_tok_offset, end_tok_offset) in zip(ner_tokens_list, ner_tokens_offsets_list):
                        if start_tok_offset >= start_offset and end_tok_offset <= end_offset:
                            sentence_tokens.append(tok)
                    sentences_tokens_list.append(sentence_tokens)
                sentences_tokens_batch.append(sentences_tokens_list)

            log.debug(f"entity_substr_batch {entity_substr_batch}")
            log.debug(f"entity_offsets_batch {entity_offsets_batch}")

            entity_substr_batch_list.append(entity_substr_batch)
            tags_batch_list.append(tags_batch)
            entity_offsets_batch_list.append(entity_offsets_batch)
            entity_probas_batch_list.append(probas_batch)
            text_len_batch_list.append([len(text) for text in text_batch])
            ner_tokens_batch_list.append(ner_tokens_batch)
            entity_positions_batch_list.append(pr_entity_positions_batch)
            sentences_tokens_batch_list.append(sentences_tokens_batch)

        doc_entity_substr_batch, doc_tags_batch, doc_entity_offsets_batch, doc_probas_batch = [], [], [], []
        doc_sentences_offsets_batch, doc_sentences_batch = [], []
        doc_ner_tokens_batch, doc_entity_positions_batch, doc_sentences_tokens_batch = [], [], []
        doc_entity_substr, doc_tags, doc_probas, doc_entity_offsets = [], [], [], []
        doc_sentences_offsets, doc_sentences = [], []
        doc_ner_tokens, doc_entity_positions, doc_sentences_tokens = [], [], []
        cur_doc_num = 0
        text_len_sum = 0
        tokens_len_sum = 0
        for entity_substr_batch, tags_batch, probas_batch, entity_offsets_batch, sentences_offsets_batch, \
            sentences_batch, text_len_batch, nums_batch, ner_tokens_batch, entity_positions_batch, sentences_tokens_batch in \
                zip(entity_substr_batch_list, tags_batch_list, entity_probas_batch_list, entity_offsets_batch_list,
                    sentences_offsets_batch_list, sentences_batch_list, text_len_batch_list, nums_batch_list,
                    ner_tokens_batch_list, entity_positions_batch_list, sentences_tokens_batch_list):
            for entity_substr, tag, probas, entity_offsets, sentences_offsets, sentences, text_len, doc_num, \
                ner_tokens, entity_positions, sentences_tokens in \
                    zip(entity_substr_batch, tags_batch, probas_batch, entity_offsets_batch, sentences_offsets_batch,
                        sentences_batch, text_len_batch, nums_batch, ner_tokens_batch, entity_positions_batch,
                        sentences_tokens_batch):
                print("entity_positions", entity_positions)
                if doc_num == cur_doc_num:
                    doc_entity_substr += entity_substr
                    doc_tags += tag
                    doc_probas += probas
                    doc_entity_offsets += [(start_offset + text_len_sum, end_offset + text_len_sum)
                                           for start_offset, end_offset in entity_offsets]
                    doc_sentences_offsets += [(start_offset + text_len_sum, end_offset + text_len_sum)
                                              for start_offset, end_offset in sentences_offsets]
                    doc_entity_positions += [[pos + tokens_len_sum for pos in entity_position] for entity_position in
                                             entity_positions]
                    doc_sentences += sentences
                    text_len_sum += text_len + 1
                    doc_ner_tokens += ner_tokens
                    tokens_len_sum += len(ner_tokens)
                    doc_sentences_tokens += sentences_tokens
                else:
                    doc_entity_substr_batch.append(doc_entity_substr)
                    doc_tags_batch.append(doc_tags)
                    doc_probas_batch.append(doc_probas)
                    doc_entity_offsets_batch.append(doc_entity_offsets)
                    doc_sentences_offsets_batch.append(doc_sentences_offsets)
                    doc_sentences_batch.append(doc_sentences)
                    doc_ner_tokens_batch.append(doc_ner_tokens)
                    doc_sentences_tokens_batch.append(doc_sentences_tokens)
                    doc_entity_substr = entity_substr
                    doc_tags = tag
                    doc_probas = probas
                    doc_entity_offsets = entity_offsets
                    doc_sentences_offsets = sentences_offsets
                    doc_entity_positions = entity_positions
                    doc_sentences = sentences
                    cur_doc_num = doc_num
                    text_len_sum = text_len
                    doc_ner_tokens = ner_tokens
                    doc_sentences_tokens = sentences_tokens
                    tokens_len_sum = len(ner_tokens)
        doc_entity_substr_batch.append(doc_entity_substr)
        doc_tags_batch.append(doc_tags)
        doc_probas_batch.append(doc_probas)
        doc_entity_offsets_batch.append(doc_entity_offsets)
        doc_sentences_offsets_batch.append(doc_sentences_offsets)
        doc_entity_positions_batch.append(doc_entity_positions)
        doc_sentences_batch.append(doc_sentences)
        doc_ner_tokens_batch.append(doc_ner_tokens)
        doc_sentences_tokens_batch.append(doc_sentences_tokens)

        return doc_entity_substr_batch, doc_entity_offsets_batch, doc_tags_batch, doc_probas_batch, \
               doc_sentences_offsets_batch, doc_sentences_batch, doc_ner_tokens_batch, doc_entity_positions_batch, doc_sentences_tokens_batch


@register('triplet_extractor')
class TripletExtractor(Component):
    def __init__(self, re_model, el_model=None, sliding_window_size=1, re_batch_size=16, **kwargs):
        self.el_model = el_model
        self.re_model = re_model
        self.sliding_window_size = sliding_window_size
        self.re_batch_size = re_batch_size

    def __call__(self, entity_substr_batch, entity_offsets_batch, tags_batch, probas_batch, sentences_offsets_batch,
                 sentences_batch, tokens_batch, entity_positions_batch, sentences_tokens_batch):
        entity_ids_batch, _ = self.el_model(entity_substr_batch, ["" for _ in entity_substr_batch],
                                            ["" for _ in entity_substr_batch])
        triplets_batch = []
        for entity_substr_list, entity_ids_list, entity_offsets_list, tags_list, probas_list, sentences_offsets_list, \
            sentences_list, tokens_list, entity_positions_list, sentences_tokens_list in \
                zip(entity_substr_batch, entity_ids_batch, entity_offsets_batch, tags_batch, probas_batch, \
                    sentences_offsets_batch, sentences_batch, tokens_batch, entity_positions_batch,
                    sentences_tokens_batch):
            used_entity_pairs = set()
            entities_samples = []
            entities_samples_substr = []
            entities_samples_ids = []
            for i in range(len(sentences_list) - self.sliding_window_size + 1):
                prev_lens = [len(sentences_tokens) for sentences_tokens in sentences_tokens_list[:i]]
                if prev_lens:
                    prev_len_sum = sum(prev_lens)
                else:
                    prev_len_sum = 0
                sentences_chunk = sentences_list[i:i + self.sliding_window_size]
                sentences_tokens_chunk = sentences_tokens_list[i:i + self.sliding_window_size]
                sentences_tokens_chunk_list = list(itertools.chain.from_iterable(sentences_tokens_chunk))

                chunk_start = sentences_offsets_list[i][0]
                chunk_end = sentences_offsets_list[i + self.sliding_window_size - 1][1]
                cur_entities = []
                for entity_substr, entity_ids, tag, (start_offset, end_offset), entity_positions in \
                        zip(entity_substr_list, entity_ids_list, tags_list, entity_offsets_list, entity_positions_list):
                    if start_offset >= chunk_start and end_offset <= chunk_end:
                        start_pos = entity_positions[0] - prev_len_sum
                        end_pos = entity_positions[-1] + 1 - prev_len_sum
                        cur_entities.append({"entity_substr": entity_substr,
                                             "entity_ids": entity_ids,
                                             "entity_offsets": (start_offset, end_offset),
                                             "entity_pos": (start_pos, end_pos),
                                             "entity_tag": tag})
                if len(cur_entities) >= 2:
                    for j in range(len(cur_entities)):
                        for k in range(len(cur_entities)):
                            if j < k:
                                subj = cur_entities[j]
                                obj = cur_entities[k]
                                if (subj["entity_offsets"], obj["entity_offsets"]) not in used_entity_pairs:
                                    entities_samples.append([sentences_tokens_chunk_list,
                                                             [[subj["entity_pos"]], [obj["entity_pos"]],
                                                              subj["entity_tag"], obj["entity_tag"]]])
                                    entities_samples_substr.append([subj["entity_substr"], obj["entity_substr"]])
                                    entities_samples_ids.append([subj["entity_ids"], obj["entity_ids"]])
                                    used_entity_pairs.add((subj["entity_offsets"], obj["entity_offsets"]))

            re_chunk_nums = len(entities_samples) // self.re_batch_size + int(
                len(entities_samples) % self.re_batch_size > 0)

            re_res_list = []
            for i in range(re_chunk_nums):
                re_chunk = entities_samples[i * self.re_batch_size:(i + 1) * self.re_batch_size]
                re_res = self.re_model(re_chunk)
                re_res_list += re_res

            triplets_list = []
            for (subj_substr, obj_substr), (subj_ids, obj_ids), re_res in \
                    zip(entities_samples_substr, entities_samples_ids, re_res_list):
                if re_res[0] not in {"Na", "no_relation"}:
                    rel_id, rel_label = re_res[0]
                    if subj_ids:
                        subj_id = subj_ids[0]
                    else:
                        subj_id = "not_in_wiki"
                    if obj_ids:
                        obj_id = obj_ids[0]
                    else:
                        obj_id = "not_in_wiki"
                    triplets_list.append([[subj_substr, rel_label, obj_substr], [subj_id, rel_id, obj_id]])

            triplets_batch.append(triplets_list)
        return triplets_batch
