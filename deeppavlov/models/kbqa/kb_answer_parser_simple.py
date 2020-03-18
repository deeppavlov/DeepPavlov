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
from typing import List, Tuple, Optional
import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.models.kbqa.kb_answer_parser_base import KBBase

log = getLogger(__name__)


@register('kb_answer_parser_wikidata')
class KBAnswerParserWikidata(KBBase):
    """
        This class generates an answer for a given question using Wikidata.
        It searches for matching triplet from the Wikidata with entity and
        relation mentioned in the question. It uses results of the Named
        Entity Recognition component to extract entity mention and Classification
        component to determine relation which connects extracted entity and the
        answer entity.
    """

    def __init__(self, top_k_classes: int,
                 debug: bool = False,
                 rule_filter_entities: bool = False,
                 language: str = "eng",
                 relations_maping_filename: Optional[str] = None,
                 templates_filename: Optional[str] = None,
                 *args, **kwargs) -> None:
        """

        Args:
            top_k_classes: number of relations with top k probabilities
            debug: whether to print debug information
            rule_filter_entities: whether to filter entities with rules
            language: russian or english
            relations_maping_filename: file with the dictionary of ids(keys) and titles(values) of relations
            from Wikidata
            templates_filename: file with the dictionary of question templates(keys) and relations for these templates
            (values)
            *args
            **kwargs
        """
        self.top_k_classes = top_k_classes
        self._debug = debug
        self.rule_filter_entities = rule_filter_entities
        self.language = language
        self._relations_filename = relations_maping_filename
        self._templates_filename = templates_filename
        super().__init__(relations_maping_filename=self._relations_filename, *args, **kwargs)

    def __call__(self, questions_batch: List[str],
                 tokens_batch: List[List[str]],
                 tags_batch: List[List[int]],
                 relations_probs_batch: List[List[float]],
                 relations_labels_batch: List[List[str]],
                 *args, **kwargs) -> List[str]:

        objects_batch = []
        confidences_batch = []

        for question, tokens, tags, relations_probs, relations_labels in \
                zip(questions_batch, tokens_batch, tags_batch, relations_probs_batch, relations_labels_batch):
            is_kbqa = self.is_kbqa_question(question, self.language)
            if is_kbqa:
                if self._templates_filename is not None:
                    entity_from_template, relations_from_template, query_type = self.template_matcher(question)
                else:
                    entity_from_template = None
                if entity_from_template:
                    relation_from_template = relations_from_template[0][0]
                    if self._debug:
                        relation_title = self._relations_mapping[relation_from_template]
                        log.debug("entity {}, relation {}".format(entity_from_template, relation_title))
                    entity_ids, entity_linking_confidences = self.linker(entity_from_template[0])
                    entity_triplets = self.extract_triplets_from_wiki(entity_ids)
                    if self.rule_filter_entities and self.language == 'rus':
                        entity_ids, entity_triplets, entity_linking_confidences = \
                            self.filter_triplets_rus(entity_triplets, entity_linking_confidences, tokens, entity_ids)

                    relation_prob = 1.0
                    obj, confidence = self.match_triplet(entity_triplets,
                                                         entity_linking_confidences,
                                                         [relation_from_template],
                                                         [relation_prob])
                else:
                    entity_from_ner = self.extract_entities(tokens, tags)
                    entity_ids, entity_linking_confidences = self.linker(entity_from_ner)
                    entity_triplets = self.extract_triplets_from_wiki(entity_ids)
                    if self.rule_filter_entities and self.language == 'rus':
                        entity_ids, entity_triplets, entity_linking_confidences = \
                            self.filter_triplets_rus(entity_triplets, entity_linking_confidences, tokens, entity_ids)

                    top_k_probs = self._parse_relations_probs(relations_probs)
                    top_k_relation_names = [self._relations_mapping[rel] for rel in relations_labels]
                    if self._debug:
                        log.debug("entity_from_ner {}, top k relations {}".format(str(entity_from_ner),
                                                                                  str(top_k_relation_names)))
                    obj, confidence = self.match_triplet(entity_triplets,
                                                         entity_linking_confidences,
                                                         relations_labels,
                                                         top_k_probs)
                objects_batch.append(obj)
                confidences_batch.append(confidence)
            else:
                objects_batch.append('')
                confidences_batch.append(0.0)

        parsed_objects_batch, confidences_batch = self.parse_wikidata_object(objects_batch, confidences_batch)

        return parsed_objects_batch, confidences_batch

    def _parse_relations_probs(self, probs: List[float]) -> List[str]:
        top_k_inds = np.asarray(probs).argsort()[-self.top_k_classes:][::-1]
        top_k_probs = [probs[k] for k in top_k_inds]
        return top_k_probs

    @staticmethod
    def extract_entities(tokens: List[str], tags: List[str]) -> str:
        entity = []
        for j, tok in enumerate(tokens):
            if tags[j] != 'O' and tags[j] != 0:
                entity.append(tok)
        entity = ' '.join(entity)

        return entity

    def extract_triplets_from_wiki(self, entity_ids: List[str]) -> List[List[List[str]]]:
        entity_triplets = []
        for entity_id in entity_ids:
            if entity_id in self.wikidata and entity_id.startswith('Q'):
                triplets_for_entity = self.wikidata[entity_id]
                entity_triplets.append(triplets_for_entity)
            else:
                entity_triplets.append([])

        return entity_triplets

    def filter_triplets_rus(self, entity_triplets: List[List[List[str]]], confidences, question_tokens: List[str],
                            srtd_cand_ent: List[Tuple[str]]) -> Tuple[
        List[Tuple[str]], List[List[List[str]]], List[float]]:

        question = ' '.join(question_tokens).lower()
        what_template = 'что '
        found_what_template = question.find(what_template) > -1
        filtered_entity_triplets = []
        filtered_entities = []
        filtered_confidences = []
        for wiki_entity, confidence, triplets_for_entity in zip(srtd_cand_ent, confidences, entity_triplets):
            entity_is_human = False
            entity_is_asteroid = False
            entity_is_named = False
            entity_title = wiki_entity
            if entity_title[0].isupper():
                entity_is_named = True
            property_is_instance_of = 'P31'
            id_for_entity_human = 'Q5'
            id_for_entity_asteroid = 'Q3863'
            for triplet in triplets_for_entity:
                if triplet[0] == property_is_instance_of and triplet[1] == id_for_entity_human:
                    entity_is_human = True
                    break
                if triplet[0] == property_is_instance_of and triplet[1] == id_for_entity_asteroid:
                    entity_is_asteroid = True
                    break
            if found_what_template and \
                    (entity_is_human or entity_is_named or entity_is_asteroid):
                continue
            filtered_entity_triplets.append(triplets_for_entity)
            filtered_entities.append(wiki_entity)
            filtered_confidences.append(confidence)

        return filtered_entities, filtered_entity_triplets, filtered_confidences
