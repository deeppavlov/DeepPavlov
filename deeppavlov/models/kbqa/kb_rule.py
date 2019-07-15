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
from io import StringIO
from logging import getLogger
from string import punctuation
from typing import List, Tuple, Optional, Dict

import nltk
import numpy as np
from udapi.block.read.conllu import Conllu
from udapi.core.node import Node
from ufal_udpipe import Model as udModel, Pipeline

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.kbqa.entity_linking import EntityLinker
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder

log = getLogger(__name__)


def descendents(node, desc_list):
    if len(node.children) > 0:
        for child in node.children:
            desc_list = descendents(child, desc_list)
    desc_list.append(node.form)

    return desc_list


@register('kb_rule')
class KBRule(Component, Serializable):
    """
        This class generates an answer for a given question using Wikidata.
        It searches for matching triplet from the Wikidata with entity and
        relation mentioned in the question. It uses templates for entity 
        and relation extraction from the question. If the question does not
        match with any of templates, syntactic parsing is used for entity
        and relation extraction.
    """

    def __init__(self, load_path: str, udpipe_filename: str, linker: EntityLinker, ft_embedder: FasttextEmbedder,
                 debug: bool = False, use_templates: bool = True, return_confidences: bool = True,
                 relations_maping_filename: str = None, templates_filename: str = None, *args, **kwargs) -> None:

        """

        Args:
            load_path: path to folder with wikidata files
            udpipe_path: path to file with udpipe model
            linker: component `deeppavlov.models.kbqa.entity_linking`
            debug: whether to print entities and relations extracted from the question
            use_templates: whether to use templates for entity and relation extraction
            relations_maping_filename: file with the dictionary of ids(keys) and titles(values) of relations
            from Wikidata
            fasstext_load_path: path to file with fasttext embeddings
            templates_filename: file with the dictionary of question templates(keys) and relations for these templates
            (values)
            return_confidences: whether to return confidences of answers
            *args:
            **kwargs:
        """
        
        super().__init__(save_path=None, load_path=load_path)
        self._debug = debug
        self.udpipe_filename = udpipe_filename
        self.use_templates = use_templates
        self.return_confidences = return_confidences
        self._relations_filename = relations_maping_filename
        self.q_to_name: Optional[Dict[str, Dict[str, str]]] = None
        self._relations_mapping: Optional[Dict[str, str]] = None
        self._templates_filename = templates_filename
        self.linker = linker
        self.ft_embedder = ft_embedder
        self.udpipe_load_path = self.load_path.parent / self.udpipe_filename
        self.ud_model = udModel.load(str(self.udpipe_load_path))
        self.full_ud_model = Pipeline(self.ud_model, "vertical", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        self.load()

    def load(self) -> None:
        with open(self.load_path, 'rb') as fl:
            self.q_to_name = pickle.load(fl)
        if self._relations_filename is not None:
            with open(self.load_path.parent / self._relations_filename, 'rb') as f:
                self._relations_mapping = pickle.load(f)
        if self._templates_filename is not None:
            with open(self.load_path.parent / self._templates_filename, 'rb') as t:
                self.templates = pickle.load(t)

    def save(self) -> None:
        pass

    def __call__(self, sentences: List[str],
                 *args, **kwargs) -> List[str]:

        objects_batch = []
        confidences_batch = []
        for sentence in sentences:
            is_kbqa = self.is_kbqa_question(sentence)
            if is_kbqa:
                q_tokens = nltk.word_tokenize(sentence)
                entity_from_template, relation_from_template = self.entities_and_rels_from_templates(q_tokens)
                if entity_from_template and self.use_templates:
                    if self._debug:
                        relation_title = self._relations_mapping[relation_from_template]
                        log.debug("using templates, entity {}, relation {}".format(entity_from_template,
                                                                                   relation_title))
                    entity_triplets, entity_linking_confidences = self.linker(entity_from_template, q_tokens)
                    rel_prob = 1.0
                    obj, confidence = self.match_triplet(entity_triplets,
                                             entity_linking_confidences,
                                             relation_from_template,
                                             rel_prob)
                    objects_batch.append(obj)
                    confidences_batch.append(confidence)
                else:
                    detected_entity, detected_rel = self.entities_and_rels_from_tree(q_tokens)
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
        
    def find_entity(self, tree, 
                          q_tokens: List[str]) -> Tuple[bool, str, str]:
        detected_entity = ""
        detected_rel = ""
        min_tree = 10
        leaf_node = None
        for node in tree.descendants:
            if len(node.children) < min_tree and node.upos in ["NOUN", "PROPN"]:
                leaf_node = node

        if leaf_node is not None:
            node = leaf_node
            desc_list = []
            entity_tokens = []
            while node.parent.upos in ["NOUN", "PROPN"] and node.parent.deprel!="root"\
                    and not node.parent.parent.form.startswith("Как"):
                node = node.parent
            detected_rel = node.parent.form
            desc_list.append(node.form)
            desc_list = descendents(node, desc_list)
            num_tok = 0
            for n, tok in enumerate(q_tokens):
                if tok in desc_list:
                    entity_tokens.append(tok)
                    num_tok = n
            if (num_tok+1) < len(q_tokens):
                if q_tokens[(num_tok+1)].isdigit():
                    entity_tokens.append(q_tokens[(num_tok+1)])
            detected_entity = ' '.join(entity_tokens)
            return True, detected_entity, detected_rel

        return False, detected_entity, detected_rel

    def find_entity_adj(self, tree: Node) -> Tuple[bool, str, str]:
        detected_rel = ""
        detected_entity = ""
        for node in tree.descendants:
            if len(node.children) <= 1 and node.upos == "ADJ":
                detected_rel = node.parent.form
                detected_entity = node.form
                return True, detected_entity, detected_rel
        
        return False, detected_entity, detected_rel

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

    def is_kbqa_question(self, question_init: str) -> bool:
        not_kbqa_question_templates = ["почему", "когда будет", "что будет", "что если", "для чего ", "как ",
                                       "что делать", "зачем", "что может"]
        kbqa_question_templates = ["как зовут", "как называется", "как звали", "как ты думаешь", "как твое мнение",
                                   "как ты считаешь"]

        question = ''.join([ch for ch in question_init if ch not in punctuation]).lower()
        is_kbqa = (all(template not in question for template in not_kbqa_question_templates) or
                   any(template in question for template in kbqa_question_templates))
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

    def entities_and_rels_from_templates(self, tokens: List[str]) -> Tuple[str, str]:
        s_sanitized = ' '.join([ch for ch in tokens if ch not in punctuation]).lower()
        s_sanitized = s_sanitized.replace("'", '').replace("`", '')
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

    def entities_and_rels_from_tree(self, q_tokens: List[str]) -> Tuple[str, str]:
        q_str = '\n'.join(q_tokens)
        s = self.full_ud_model.process(q_str)
        tree = Conllu(filehandle=StringIO(s)).read_tree()
        fnd, detected_entity, detected_rel = self.find_entity(tree, q_tokens)
        if fnd == False:
            fnd, detected_entity, detected_rel = self.find_entity_adj(tree)
        detected_entity = detected_entity.replace("первый ", '')
        return detected_entity, detected_rel

    def match_triplet(self,
                      entity_triplets: List[List[List[str]]],
                      entity_linking_confidences: List[float],
                      relation: str,
                      rel_prob: float) -> Tuple[str, float]:
        obj = ''
        confidence = 0.0
        for entities, linking_confidence in zip(entity_triplets, entity_linking_confidences):
            for rel_triplets in entities:
                relation_from_wiki = rel_triplets[0]
                if relation == relation_from_wiki:
                    obj = rel_triplets[1]
                    confidence = linking_confidence * rel_prob
                    return obj, confidence
        return obj, confidence

