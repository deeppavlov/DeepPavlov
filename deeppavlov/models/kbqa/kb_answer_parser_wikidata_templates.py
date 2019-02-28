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

import numpy as np
import pickle
from deeppavlov.core.models.serializable import Serializable

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from pathlib import Path
from datetime import datetime
from string import punctuation
from deeppavlov.models.kbqa.entity_linking import EntityLinker

log = getLogger(__name__)


@register('kb_answer_parser_wikidata_templates')
class KBAnswerParserWikidata(Component, Serializable):
    """
       Class for generation of answer using triplets with the entity
       in the question and relations predicted from the question by the
       relation prediction model.
       We search a triplet with the predicted relations
    """

    def __init__(self, load_path: str, top_k_classes: int, classes_vocab_keys: Tuple,
                 debug: bool = False, relations_maping_filename=None, entities_filename=None, wiki_filename=None, templates_filename=None, *args, **kwargs) -> None:
        super().__init__(save_path=None, load_path=load_path)
        self.top_k_classes = top_k_classes
        self.classes = list(classes_vocab_keys)
        self._debug = debug
        self._relations_filename = relations_maping_filename
        self._entities_filename = entities_filename
        self._wiki_filename = wiki_filename
        self._templates_filename = templates_filename
        self._q_to_name = None
        self._relations_mapping = None
        self.name_to_q = None
        self.wikidata = None
        self.templates = None
        self.load()
        self.linker = EntityLinker(self.name_to_q, self.wikidata)

    def load(self) -> None:
        load_path = Path(self.load_path).expanduser()
        with open(load_path, 'rb') as fl:
            self._q_to_name = pickle.load(fl)
        if self._relations_filename is not None:
            with open(self.load_path.parent / self._relations_filename, 'rb') as f:
                self._relations_mapping = pickle.load(f)
        with open(self.load_path.parent / self._entities_filename, 'rb') as e:
            self.name_to_q = pickle.load(e)
        with open(self.load_path.parent / self._wiki_filename, 'rb') as w:
            self.wikidata = pickle.load(w)
        with open(self.load_path.parent / self._templates_filename, 'rb') as t:
            self.templates = pickle.load(t)

    def save(self):
        pass

    def __call__(self, tokens_batch: List[List[str]],
                 tags_batch: List[List[int]],
                 relations_probs_batch: List[List[str]],
                 *args, **kwargs) -> List[str]:

        objects_batch = []
        for tokens, tags, relations_probs in zip(tokens_batch, tags_batch, relations_probs_batch):
            entity, relation = self.entities_and_rels_from_templates(tokens)
            if entity:
                entity_triplets, confidences = self.linker(entity)
                found = False
                for n, entities in enumerate(entity_triplets):
                    for rel_triplets in entities:
                        relation_from_wiki = rel_triplets[0]
                        if relation == relation_from_wiki:
                            obj = rel_triplets[1]
                            found = True
                            break
                    if found or n == 5:
                        break
                if not found:
                    obj = ''
                objects_batch.append(obj)

            if not entity:
                entity = self.extract_entities(tokens, tags)
                if not entity:
                    objects_batch.append('')
                else:
                    entity_triplets, confidences = self.linker(entity)
                    relations = self._parse_relations_probs(relations_probs)

                    found = False
                    for predicted_relation, rel_prob in zip(relations, relations_probs):
                        for n, entities in enumerate(entity_triplets):
                            for rel_triplets in entities:
                                relation_from_wiki = rel_triplets[0]
                                if predicted_relation == relation_from_wiki:
                                    obj = rel_triplets[1]
                                    found = True
                                    break
                            if found or n == 5:
                                break
                        if found:
                            break
                    if not found:
                        obj = ''
                    objects_batch.append(obj)

        word_batch = []

        for n, obj in enumerate(objects_batch):
            if len(obj) > 0:
                if obj.startswith('Q'):
                    if obj in self._q_to_name:
                        word = self._q_to_name[obj]["name"]
                        word_batch.append(word)
                    else:
                        word_batch.append('Not Found')
                elif obj.count('-') == 2 and int(obj.split('-')[0]) > 1000:
                    dt = datetime.strptime(obj, "%Y-%m-%d")
                    obj = dt.strftime("%d %B %Y")
                    word_batch.append(obj)
                else:
                    word_batch.append(obj)
            else:
                word_batch.append('Not Found')

        return word_batch

    def _parse_relations_probs(self, probas: List[float]) -> List[str]:
        top_k_inds = np.asarray(probas).argsort()[-self.top_k:][::-1]  # Make it top n and n to the __init__
        top_k_classes = [self.classes[k] for k in top_k_inds]

        return top_k_classes

    def extract_entities(tokens, tags):
        entity = []
        for j, tok in enumerate(tokens):
            if tags[j] != 0:
                entity.append(tok)
        entity = ' '.join(entity)

        return entity

    def entities_and_rels_from_templates(self, tokens):
        s_sanitized = ' '.join([ch for ch in tokens if ch not in punctuation]).lower()
        ent = ''
        relation = ''
        for template in self.templates:
            template_start, template_end = template.lower().split('xxx')
            if s_sanitized.startswith(template_start) and s_sanitized.endswith(template_end):
                ent_cand = s_sanitized[len(template_start): -len(template_end) or len(s_sanitized)]
                if len(ent_cand) < len(ent) or len(ent) == 0:
                    ent = ent_cand
                    relation = self.templates[template]

        return ent, relation
