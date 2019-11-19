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

import pickle
from pathlib import Path
from string import punctuation
from logging import getLogger
from typing import List, Tuple, Optional, Dict

import numpy as np

from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.models.kbqa.entity_linking import EntityLinker

log = getLogger(__name__)


@register('kb_answer_parser_wikidata')
class KBAnswerParserWikidata(Component, Serializable):
    """
        This class generates an answer for a given question using Wikidata.
        It searches for matching triplet from the Wikidata with entity and
        relation mentioned in the question. It uses results of the Named
        Entity Recognition component to extract entity mention and Classification
        component to determine relation which connects extracted entity and the
        answer entity.
    """

    def __init__(self, load_path: str, top_k_classes: int, linker: EntityLinker, classes_vocab_keys: Tuple,
                 debug: bool = False, relations_maping_filename: str = None, templates_filename: str = None,
                 return_confidences: bool = True, *args, **kwargs) -> None:
        """

        Args:
            load_path: path to folder with wikidata files
            top_k_classes: number of relations with top k probabilities
            linker: component `deeppavlov.models.kbqa.entity_linking`
            classes_vocab_keys: list of relations predicted by `deeppavlov.models.ner.network` model
            debug: whether to print entities and relations extracted from the question
            relations_maping_filename: file with the dictionary of ids(keys) and titles(values) of relations
            from Wikidata
            templates_filename: file with the dictionary of question templates(keys) and relations for these templates
            (values)
            return_confidences: whether to return confidences of answers
            *args:
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self.top_k_classes = top_k_classes
        self.classes = list(classes_vocab_keys)
        self._debug = debug
        self._relations_filename = relations_maping_filename
        self._templates_filename = templates_filename
        self._q_to_name: Optional[Dict[str, Dict[str, str]]] = None
        self._relations_mapping: Optional[Dict[str, str]] = None
        self.templates: Optional[Dict[str, str]] = None
        self.return_confidences = return_confidences
        self.linker = linker
        self.load()

    def load(self) -> None:
        with open(self.load_path, 'rb') as fl:
            self._q_to_name = pickle.load(fl)
        if self._relations_filename is not None:
            with open(self.load_path.parent / self._relations_filename, 'rb') as f:
                self._relations_mapping = pickle.load(f)
        if self._templates_filename is not None:
            with open(self.load_path.parent / self._templates_filename, 'rb') as t:
                self.templates = pickle.load(t)

    def save(self) -> None:
        pass

    def __call__(self, tokens_batch: List[List[str]],
                 tags_batch: List[List[int]],
                 relations_probs_batch: List[List[float]],
                 *args, **kwargs) -> List[str]:

        objects_batch = []
        confidences_batch = []

        for tokens, tags, relations_probs in zip(tokens_batch, tags_batch, relations_probs_batch):
            is_kbqa = self.is_kbqa_question(tokens)
            if is_kbqa:
                if self._templates_filename is not None:
                    entity_from_template, relation_from_template = self.entities_and_rels_from_templates(tokens)
                else:
                    entity_from_template = None
                if entity_from_template:
                    if self._debug:
                        relation_title = self._relations_mapping[relation_from_template]
                        log.debug("entity {}, relation {}".format(entity_from_template, relation_title))
                    entity_triplets, entity_linking_confidences = self.linker(entity_from_template, tokens)
                    relation_prob = 1.0
                    obj, confidence = self._match_triplet(entity_triplets,
                                                          entity_linking_confidences,
                                                          [relation_from_template],
                                                          [relation_prob])
                else:
                    entity_from_ner = self.extract_entities(tokens, tags)
                    entity_triplets, entity_linking_confidences = self.linker(entity_from_ner, tokens)
                    top_k_relations, top_k_probs = self._parse_relations_probs(relations_probs)
                    top_k_relation_names = [self._relations_mapping[rel] for rel in top_k_relations]
                    if self._debug:
                        log.debug("top k relations {}" .format(str(top_k_relation_names)))
                    obj, confidence = self._match_triplet(entity_triplets,
                                                          entity_linking_confidences,
                                                          top_k_relations,
                                                          top_k_probs)
                objects_batch.append(obj)
                confidences_batch.append(confidence)
            else:
                objects_batch.append('')
                confidences_batch.append(0.0)

        parsed_objects_batch, confidences_batch = self._parse_wikidata_object(objects_batch, confidences_batch)
        if self.return_confidences:
            return parsed_objects_batch, confidences_batch
        else:
            return parsed_objects_batch

    def _parse_wikidata_object(self,
                               objects_batch: List[str],
                               confidences_batch: List[float]) -> Tuple[List[str], List[float]]:
        parsed_objects = []
        for n, obj in enumerate(objects_batch):
            if len(obj) > 0:
                if obj.startswith('Q'):
                    if obj in self._q_to_name:
                        parsed_object = self._q_to_name[obj]["name"]
                        parsed_objects.append(parsed_object)
                    else:
                        parsed_objects.append('Not Found')
                        confidences_batch[n] = 0.0
                else:
                    parsed_objects.append(obj)
            else:
                parsed_objects.append('Not Found')
                confidences_batch[n] = 0.0
        return parsed_objects, confidences_batch

    @staticmethod
    def _match_triplet(entity_triplets: List[List[str]],
                       entity_linking_confidences: List[float],
                       relations: List[int],
                       relations_probs: List[float]) -> Tuple[str, float]:
        obj = ''
        confidence = 0.0
        for predicted_relation, rel_prob in zip(relations, relations_probs):
            for entities, linking_confidence in zip(entity_triplets, entity_linking_confidences):
                for rel_triplets in entities:
                    relation_from_wiki = rel_triplets[0]
                    if predicted_relation == relation_from_wiki:
                        obj = rel_triplets[1]
                        confidence = linking_confidence * rel_prob
                        return obj, confidence
        return obj, confidence

    def _parse_relations_probs(self, probs: List[float]) -> Tuple[List[str], List[str]]:
        top_k_inds = np.asarray(probs).argsort()[-self.top_k_classes:][::-1]
        top_k_classes = [self.classes[k] for k in top_k_inds]
        top_k_probs = [probs[k] for k in top_k_inds]
        return top_k_classes, top_k_probs

    @staticmethod
    def extract_entities(tokens: List[str], tags: List[str]) -> str:
        entity = []
        for j, tok in enumerate(tokens):
            if tags[j] != 0:  # TODO: replace with tag 'O' (not necessary 0)
                entity.append(tok)
        entity = ' '.join(entity)

        return entity

    def entities_and_rels_from_templates(self, tokens: List[List[str]]) -> Tuple[str, int]:
        s_sanitized = ' '.join([ch for ch in tokens if ch not in punctuation]).lower()
        ent = ''
        relation = ''
        for template in self.templates:
            template_start, template_end = template.lower().split('xxx')
            if template_start in s_sanitized and template_end in s_sanitized:
                template_start_pos = s_sanitized.find(template_start)
                template_end_pos = s_sanitized.find(template_end)
                ent_cand = s_sanitized[template_start_pos+len(template_start): template_end_pos or len(s_sanitized)]
                if len(ent_cand) < len(ent) or len(ent) == 0:
                    ent = ent_cand
                    relation = self.templates[template]
        return ent, relation

    def is_kbqa_question(self, question_tokens: List[List[str]]) -> bool:
        not_kbqa_question_templates = ["почему", "когда будет", "что будет", "что если", "для чего ", "как ", \
                                       "что делать", "зачем", "что может"]
        kbqa_question_templates = ["как зовут", "как называется", "как звали", "как ты думаешь", "как твое мнение", \
                                   "как ты считаешь"]
        question_init = ' '.join(question_tokens)
        question = ''.join([ch for ch in question_init if ch not in punctuation]).lower()
        is_kbqa = (all(template not in question for template in not_kbqa_question_templates) or
                   all(template in question for template in kbqa_question_templates))
        return is_kbqa
