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
from string import punctuation
from typing import List, Tuple, Optional, Dict

from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.kbqa.entity_linking import EntityLinker
from deeppavlov.models.kbqa.template_matcher import TemplateMatcher


class KBBase(Component, Serializable):
    """
        Base class to generate an answer for a given question using Wikidata.
    """

    def __init__(self, load_path: str, wiki_filename: str, linker: EntityLinker,
                 template_matcher: TemplateMatcher, q2name_filename: str = None,
                 relations_maping_filename: Optional[str] = None,
                 *args, **kwargs) -> None:

        """

        Args:
            load_path: path to folder with wikidata files
            linker: component `deeppavlov.models.kbqa.entity_linking`
            template_matcher: component `deeppavlov.models.kbqa.template_matcher`
            relations_maping_filename: file with the dictionary of ids(keys) and titles(values) of relations
            from Wikidata
            templates_filename: file with the dictionary of question templates(keys) and relations for these templates
            (values)
            *args:
            **kwargs:
        """

        super().__init__(save_path=None, load_path=load_path)
        self._relations_filename = relations_maping_filename
        self.wiki_filename = wiki_filename
        self.q2name_filename = q2name_filename
        self.q_to_name: Optional[Dict[str, Dict[str, str]]] = None
        self._relations_mapping: Optional[Dict[str, str]] = None
        self.linker = linker
        self.template_matcher = template_matcher
        self.load()

    def load(self) -> None:
        with open(self.load_path / self.q2name_filename, 'rb') as fl:
            self.q_to_name = pickle.load(fl)
        if self._relations_filename is not None:
            with open(self.load_path / self._relations_filename, 'rb') as f:
                self._relations_mapping = pickle.load(f)
        with open(self.load_path / self.wiki_filename, 'rb') as fl:
            self.wikidata = pickle.load(fl)

    def save(self) -> None:
        pass

    def is_kbqa_question(self, question_init: str, lang: str) -> bool:
        is_kbqa = True
        not_kbqa_question_templates_rus = ["почему", "когда будет", "что будет", "что если", "для чего ", "как ",
                                           "что делать", "зачем", "что может"]
        not_kbqa_question_templates_eng = ["why", "what if", "how"]
        kbqa_question_templates_rus = ["как зовут", "как называется", "как звали", "как ты думаешь", "как твое мнение",
                                       "как ты считаешь"]

        question = ''.join([ch for ch in question_init if ch not in punctuation]).lower()
        if lang == "rus":
            is_kbqa = (all(template not in question for template in not_kbqa_question_templates_rus) or
                       any(template in question for template in kbqa_question_templates_rus))
        if lang == "eng":
            is_kbqa = all(template not in question for template in not_kbqa_question_templates_eng)
        return is_kbqa

    def parse_wikidata_object(self,
                              objects_batch: List[str],
                              confidences_batch: List[float]) -> Tuple[List[str], List[float]]:
        parsed_objects = []
        for n, obj in enumerate(objects_batch):
            if len(obj) > 0:
                if obj.startswith('Q'):
                    if obj in self.q_to_name:
                        parsed_object = self.q_to_name[obj]["name"]
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

    def match_triplet(self,
                      entity_triplets: List[List[List[str]]],
                      entity_linking_confidences: List[float],
                      relations: List[str],
                      relation_probs: List[float]) -> Tuple[str, float]:
        obj = ''
        confidence = 0.0
        for predicted_relation, rel_prob in zip(relations, relation_probs):
            for entities, linking_confidence in zip(entity_triplets, entity_linking_confidences):
                for rel_triplets in entities:
                    relation_from_wiki = rel_triplets[0]
                    if predicted_relation == relation_from_wiki:
                        obj = rel_triplets[1]
                        confidence = linking_confidence * rel_prob
                        return obj, confidence
        return obj, confidence
