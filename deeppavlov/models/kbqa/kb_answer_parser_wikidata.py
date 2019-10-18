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

    def __init__(self, top_k_classes: int, classes_vocab_keys: Tuple, debug: bool = False,
                 relations_maping_filename: Optional[str] = None, templates_filename: Optional[str] = None,
                 *args, **kwargs) -> None:
        """

        Args:
            top_k_classes: number of relations with top k probabilities
            classes_vocab_keys: list of relations predicted by `deeppavlov.models.ner.network` model
            debug: whether to print entities and relations extracted from the question
            relations_maping_filename: file with the dictionary of ids(keys) and titles(values) of relations
            from Wikidata
            templates_filename: file with the dictionary of question templates(keys) and relations for these templates
            (values)
            *args:
            **kwargs:
        """
        self.top_k_classes = top_k_classes
        self.classes = list(classes_vocab_keys)
        self._debug = debug
        self._relations_filename = relations_maping_filename
        self._templates_filename = templates_filename
        super().__init__(relations_maping_filename=self._relations_filename, *args, **kwargs)

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
                    entity_from_template, relation_from_template = self.template_matcher(tokens)
                else:
                    entity_from_template = None
                if entity_from_template:
                    if self._debug:
                        relation_title = self._relations_mapping[relation_from_template]
                        log.debug("entity {}, relation {}".format(entity_from_template, relation_title))
                    entity_triplets, entity_linking_confidences = self.linker(entity_from_template, tokens)
                    relation_prob = 1.0
                    obj, confidence = self.match_triplet(entity_triplets,
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
                    obj, confidence = self.match_triplet(entity_triplets,
                                                         entity_linking_confidences,
                                                         top_k_relations,
                                                         top_k_probs)
                objects_batch.append(obj)
                confidences_batch.append(confidence)
            else:
                objects_batch.append('')
                confidences_batch.append(0.0)

        parsed_objects_batch, confidences_batch = self.parse_wikidata_object(objects_batch, confidences_batch)

        return parsed_objects_batch, confidences_batch

    def _parse_relations_probs(self, probs: List[float]) -> Tuple[List[str], List[str]]:
        top_k_inds = np.asarray(probs).argsort()[-self.top_k_classes:][::-1]
        top_k_classes = [self.classes[k] for k in top_k_inds]
        top_k_probs = [probs[k] for k in top_k_inds]
        return top_k_classes, top_k_probs

    @staticmethod
    def extract_entities(tokens: List[str], tags: List[str]) -> str:
        entity = []
        for j, tok in enumerate(tokens):
            if tags[j] != 'O':
                entity.append(tok)
        entity = ' '.join(entity)

        return entity

