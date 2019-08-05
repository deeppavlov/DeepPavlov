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
from typing import List, Tuple

import nltk
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.models.kbqa.tree_parser import TreeParser
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.kbqa.kb_answer_parser_base import KBBase

log = getLogger(__name__)


@register('kb_tree')
class KBTree(KBBase):
    """
        This class generates an answer for a given question using Wikidata.
        It searches for matching triplet from the Wikidata with entity and
        relation mentioned in the question. It uses templates for entity 
        and relation extraction from the question. If the question does not
        match with any of templates, syntactic parsing is used for entity
        and relation extraction.
    """

    def __init__(self, tree_parser: TreeParser, ft_embedder: FasttextEmbedder,
                 debug: bool = False, use_templates: bool = True, return_confidences: bool = True,
                     *args, **kwargs) -> None:

        """

        Args:
            tree_parser: component `deeppavlov.models.kbqa.tree_parser`
            ft_embedder: component `deeppavlov.models.embedders.fasttext`
            debug: whether to print entities and relations extracted from the question
            use_templates: whether to use templates for entity and relation extraction
            return_confidences: whether to return confidences of answers
            *args:
            **kwargs:
        """
        
        super().__init__(*args, **kwargs)
        self._debug = debug
        self.use_templates = use_templates
        self.return_confidences = return_confidences
        self.tree_parser = tree_parser
        self.ft_embedder = ft_embedder

    def __call__(self, sentences: List[str],
                 *args, **kwargs) -> List[str]:

        objects_batch = []
        confidences_batch = []
        for sentence in sentences:
            is_kbqa = self.is_kbqa_question(sentence)
            if is_kbqa:
                q_tokens = nltk.word_tokenize(sentence)
                entity_from_template, relation_from_template = self.template_matcher(q_tokens)
                if entity_from_template and self.use_templates:
                    if self._debug:
                        relation_title = self._relations_mapping[relation_from_template]["name"]
                        log.debug("using templates, entity {}, relation {}".format(entity_from_template,
                                                                                   relation_title))
                    entity_triplets, entity_linking_confidences = self.linker(entity_from_template, q_tokens)
                    rel_prob = 1.0
                    obj, confidence = self.match_triplet(entity_triplets,
                                             entity_linking_confidences,
                                             [relation_from_template],
                                             [rel_prob])
                    objects_batch.append(obj)
                    confidences_batch.append(confidence)
                else:
                    detected_entity, detected_rel = self.tree_parser(q_tokens)
                    if detected_entity:
                        if self._debug:
                            log.debug("using syntactic tree, entity {}, relation {}".format(detected_entity,
                                                                                            detected_rel))
                        entity_triplets, entity_linking_confidences = self.linker(detected_entity, q_tokens)
                        obj, confidence = self.match_rel(entity_triplets, entity_linking_confidences,
                                                         detected_rel, sentence)
                        objects_batch.append(obj)
                        confidences_batch.append(confidence)
                    else:
                        objects_batch.append('')
                        confidences_batch.append(0.0)
            else:
                objects_batch.append('')
                confidences_batch.append(0.0)

        parsed_objects_batch, confidences_batch = self.parse_wikidata_object(objects_batch, confidences_batch)

        if self.return_confidences:
            return parsed_objects_batch, confidences_batch
        else:
            return parsed_objects_batch

    def filter_triplets(self, triplets: List[List[str]],
                              sentence: str) -> List[List[str]]:
        where_rels = ["P131", "P30", "P17", "P276", "P19", "P20", "P119"]
        when_rels = ["P585", "P569", "P570", "P571", "P580", "P582"]
        filtered_triplets = []
        where_templates = ["где"]
        when_templates = ["когда", "дата", "дату", "в каком году", "год"]
        fl_where_question = False
        fl_when_question = False
        for template in when_templates:
            if template in sentence.lower():
                fl_when_question = True
                break
        for template in where_templates:
            if template in sentence.lower():
                fl_where_question = True
                break

        if fl_when_question:
            for triplet in triplets:
                rel_id = triplet[0]
                if rel_id in when_rels:
                    filtered_triplets.append(triplet)
        elif fl_where_question:
            for triplet in triplets:
                rel_id = triplet[0]
                if rel_id in where_rels:
                    filtered_triplets.append(triplet)
        else:
            filtered_triplets = triplets

        return filtered_triplets

    """
    method which calculates cosine similarity between average fasttext embedding of
    tokens of relation extracted from syntactic tree and all relations from wikidata
    and we find which relation from wikidata has the biggest cosine similarity
    """
    def match_rel(self,
                  entity_triplets: List[List[List[str]]],
                  entity_linking_confidences: List[float],
                  detected_rel: str,
                  sentence: str) -> Tuple[str, float]:
        av_detected_emb = self.av_emb(detected_rel)
        max_score = 0.0
        found_obj = ""
        confidence = 0.0
        entity_triplets_flat = [item for sublist in entity_triplets for item in sublist]
        filtered_triplets = self.filter_triplets(entity_triplets_flat, sentence)
        for triplets, linking_confidence in zip(entity_triplets, entity_linking_confidences):
            for triplet in triplets:
                scores = []
                rel_id = triplet[0]
                obj = triplet[1]
                if rel_id in self._relations_mapping and triplet in filtered_triplets:
                    rel_name = self._relations_mapping[rel_id]["name"]
                    if rel_name == detected_rel:
                        found_obj = obj
                        rel_prob = 1.0
                        confidence = linking_confidence * rel_prob
                        return found_obj, confidence
                    else:
                        name_emb = self.av_emb(rel_name)
                        scores.append(np.dot(av_detected_emb, name_emb))
                        if "aliases" in self._relations_mapping[rel_id]:
                            rel_aliases = self._relations_mapping[rel_id]["aliases"]
                            for alias in rel_aliases:
                                if alias == detected_rel:
                                    found_obj = obj
                                    rel_prob = 1.0
                                    confidence = linking_confidence * rel_prob
                                    return found_obj, confidence
                                else:
                                    alias_emb = self.av_emb(alias)
                                    scores.append(np.dot(av_detected_emb, alias_emb))
                        if np.asarray(scores).mean() > max_score:
                            max_score = np.asarray(scores).mean()
                            rel_prob = min(max_score/10.0, 1.0)
                            confidence = linking_confidence * rel_prob
                            found_obj = obj

        return found_obj, confidence

    def av_emb(self, rel: str) -> List[float]:
        rel_tokens = nltk.word_tokenize(rel)
        emb = []
        for tok in rel_tokens:
            emb.append(self.ft_embedder._get_word_vector(tok))
        av_emb = np.asarray(emb).mean(axis=0)
        return av_emb

