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
from collections import defaultdict
from logging import getLogger
from string import punctuation
from typing import List, Tuple, Union, Any

import numpy as np
from nltk.corpus import stopwords

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)
punctuation = punctuation.replace('+', '')


@register('question_sign_checker')
class QuestionSignChecker:
    def __init__(self, delete_brackets: bool = False, **kwargs):
        self.delete_brackets = delete_brackets
        self.replace_tokens = [(" '", ' "'), ("' ", '" '), (" ?", "?"), ("  ", " ")]

    def __call__(self, questions: List[str]) -> List[str]:
        """Adds question sign if it is absent or replaces dots in the end with question sign."""
        questions_clean = []
        for question in questions:
            question = question if question.endswith('?') else f'{question.rstrip(".")}?'
            if self.delete_brackets:
                brackets_text = re.findall(r"(\(.*?\))", question)
                for elem in brackets_text:
                    question = question.replace(elem, " ")
            for old_tok, new_tok in self.replace_tokens:
                question = question.replace(old_tok, new_tok)
            questions_clean.append(question)
        return questions_clean


@register('entity_type_split')
def entity_type_split(entities_batch: List[List[str]], tags_batch: List[List[str]]) -> Tuple[
    List[List[str]], List[List[str]], List[List[str]]]:
    f_entities_batch, f_types_batch, f_tags_batch = [], [], []
    for entities_list, tags_list in zip(entities_batch, tags_batch):
        f_entities_list, f_types_list, f_tags_list = [], [], []
        for entity, tag in zip(entities_list, tags_list):
            if tag != "T":
                f_entities_list.append(entity)
                f_tags_list.append(tag.lower())
            else:
                f_types_list.append(entity)
        f_entities_batch.append(f_entities_list)
        f_tags_batch.append(f_tags_list)
        f_types_batch.append(f_types_list)
    return f_entities_batch, f_tags_batch, f_types_batch


@register('entity_detection_parser')
class EntityDetectionParser(Component):
    """This class parses probabilities of tokens to be a token from the entity substring."""

    def __init__(self, o_tag: str, tags_file: str, entity_tags: List[str] = None, ignore_points: bool = False,
                 thres_proba: float = 0.8, make_tags_from_probas: bool = False, lang: str = "en",
                 ignored_tags: List[str] = None, **kwargs):
        """
        Args:
            o_tag: tag for tokens which are neither entities nor types
            tags_file: filename with NER tags
            entity_tags: tags for entities
            ignore_points: whether to consider points as separate symbols
            thres_proba: if the probability of the tag is less than thres_proba, we assign the tag as 'O'
            make_tags_from_probas: whether to define token tags from confidences from sequence tagging model
            lang: language of texts
            ignored_tags: not used tags of entities
        """
        self.entity_tags = entity_tags
        self.o_tag = o_tag
        self.ignore_points = ignore_points
        self.thres_proba = thres_proba
        self.tag_ind_dict = {}
        with open(str(expand_path(tags_file))) as fl:
            tags = [line.split('\t')[0] for line in fl.readlines()]
            self.tags = tags
            if self.entity_tags is None:
                self.entity_tags = list(
                    {tag.split('-')[1] for tag in tags if len(tag.split('-')) > 1}.difference({self.o_tag}))

            self.entity_prob_ind = {entity_tag: [i for i, tag in enumerate(tags) if entity_tag in tag]
                                    for entity_tag in self.entity_tags}
            self.tags_ind = {tag: i for i, tag in enumerate(tags)}
            self.et_prob_ind = [i for tag, ind in self.entity_prob_ind.items() for i in ind]
            for entity_tag, tag_ind in self.entity_prob_ind.items():
                for ind in tag_ind:
                    self.tag_ind_dict[ind] = entity_tag
            self.tag_ind_dict[0] = self.o_tag
        self.make_tags_from_probas = make_tags_from_probas
        if lang == "en":
            self.stopwords = set(stopwords.words("english"))
        elif lang == "ru":
            self.stopwords = set(stopwords.words("russian"))
        else:
            raise ValueError(f'Unsupported lang value: "{lang}". Only "en" and "ru" are allowed.')
        self.ignored_tags = ignored_tags or []

    def __call__(self, question_tokens_batch: List[List[str]], tokens_info_batch: List[List[List[float]]],
                 tokens_probas_batch: np.ndarray) -> \
            Tuple[List[dict], List[dict], List[dict]]:
        """
        Args:
            question_tokens_batch: tokenized questions
            tokens_info_batch: list of tags of question tokens
            tokens_probas_probas: list of probabilities of question tokens
        Returns:
            Batch of dicts where keys are tags and values are substrings corresponding to tags
            Batch of substrings which correspond to entity types
            Batch of lists of token indices in the text which correspond to entities
        """
        entities_batch = []
        positions_batch = []
        probas_batch = []
        for tokens, tags, probas in \
                zip(question_tokens_batch, tokens_info_batch, tokens_probas_batch):
            if self.make_tags_from_probas:
                tags, _ = self.tags_from_probas(tokens, probas)
            tags = self.correct_quotes(tokens, tags, probas)
            tags = self.correct_tags(tokens, tags)
            entities, positions, entities_probas = self.entities_from_tags(tokens, tags, probas)
            entities_batch.append(entities)
            positions_batch.append(positions)
            probas_batch.append(entities_probas)
        return entities_batch, positions_batch, probas_batch

    def tags_from_probas(self, tokens: List[str], probas: np.array) -> Tuple[List[Union[str, List[str]]], List[Any]]:
        """
        This method makes a list of tags from a list of probas for tags
        Args:
            tokens: text tokens list
            probas: probabilities for tokens to belong to particular tags
        Returns:
            list of tags for tokens
            list of probabilities of these tags
        """
        tags = []
        tag_probas = []
        for token, proba in zip(tokens, probas):
            if proba[0] < self.thres_proba:
                tag_num = np.argmax(proba[1:]) + 1
            else:
                tag_num = 0
            tags.append(self.tags[tag_num])
            tag_probas.append(proba[tag_num])

        return tags, tag_probas

    def correct_tags(self, tokens: List[str], tags: List[str]) -> List[str]:
        for i in range(len(tags) - 2):
            if len(tags[i]) > 1 and tags[i].startswith("B-"):
                tag = tags[i].split("-")[1]
                if tags[i + 2] == f"I-{tag}" and tags[i + 1] != f"I-{tag}":
                    tags[i + 1] = f"I-{tag}"
            if tokens[i + 1] in '«' and tags[i] != "O":
                tags[i] = "O"
                tags[i + 1] = "O"
            if len(tags[i]) > 1 and tags[i].split("-")[1] == "EVENT":
                found_n = -1
                for j in range(i + 1, i + 3):
                    if re.findall(r"[\d]{3,4}", tokens[j]):
                        found_n = j
                        break
                if found_n > 0:
                    for j in range(i + 1, found_n + 1):
                        tags[j] = "I-EVENT"
            if i < len(tokens) - 3 and len(tokens[i]) == 1 and tokens[i + 1] == "." and len(tokens[i + 2]) == 1 \
                    and tokens[i + 3] == "." and tags[i + 2].startswith("B-"):
                tag = tags[i + 2].split("-")[1]
                tags[i] = f"B-{tag}"
                tags[i + 1] = f"I-{tag}"
                tags[i + 2] = f"I-{tag}"
        return tags

    def correct_quotes(self, tokens: List[str], tags: List[str], probas: np.array) -> List[str]:
        quotes = {"«": "»", '"': '"'}
        for i in range(len(tokens)):
            if tokens[i] in {"«", '"'}:
                quote_start = tokens[i]
                end_pos = 0
                for j in range(i + 1, len(tokens)):
                    if tokens[j] == quotes[quote_start]:
                        end_pos = j
                        break
                if end_pos and end_pos != i + 1:
                    probas_sum = np.sum(probas[i + 1:end_pos], axis=0)
                    tags_probas = {}
                    for tag in self.entity_prob_ind:
                        for ind in self.entity_prob_ind[tag]:
                            if tag not in tags_probas:
                                tags_probas[tag] = probas_sum[ind]
                            else:
                                tags_probas[tag] += probas_sum[ind]
                    tags_probas = list(tags_probas.items())
                    tags_probas = sorted(tags_probas, key=lambda x: x[1], reverse=True)
                    found_tag = ""
                    for tag, _ in tags_probas:
                        if tag != "PERSON":
                            found_tag = tag
                            break
                    if found_tag:
                        tags[i + 1] = f"B-{found_tag}"
                        for j in range(i + 2, end_pos):
                            tags[j] = f"I-{found_tag}"
        return tags

    def add_entity(self, entity: str, c_tag: str) -> None:
        replace_tokens = [(' - ', '-'), ("'s", ''), (' .', '.'), ('{', ''), ('}', ''),
                          ('  ', ' '), ('"', "'"), ('(', ''), (')', ''), (' +', '+')]
        if entity and (entity[-1] in punctuation or entity[-1] == "»"):
            entity = entity[:-1]
            self.ent_pos_dict[c_tag] = self.ent_pos_dict[c_tag][:-1]
        if entity and (entity[0] in punctuation or entity[0] == "«"):
            entity = entity[1:]
            self.ent_pos_dict[c_tag] = self.ent_pos_dict[c_tag][1:]
        entity = ' '.join(entity)
        for old, new in replace_tokens:
            entity = entity.replace(old, new)
        if entity and entity.lower() not in self.stopwords:
            cur_probas = self.ent_probas_dict[c_tag]
            self.ents_pos_probas_dict[c_tag].append((entity, self.ent_pos_dict[c_tag],
                                                     round(sum(cur_probas) / len(cur_probas), 4)))
        self.ent_dict[c_tag] = []
        self.ent_pos_dict[c_tag] = []
        self.ent_probas_dict[c_tag] = []

    def entities_from_tags(self, tokens: List[str], tags: List[str],
                                 tag_probas: List[List[float]]) -> Tuple[dict, dict, dict]:
        """
        This method makes lists of substrings corresponding to entities and entity types
        and a list of indices of tokens which correspond to entities
        Args:
            tokens: list of tokens of the text
            tags: list of tags for tokens
            tag_probas: list of probabilities of tags
        Returns:
            list of entity substrings (or a dict of tags (keys) and entity substrings (values))
            list of substrings for entity types
            list of indices of tokens which correspond to entities (or a dict of tags (keys)
                and list of indices of entity tokens)
        """
        self.ent_dict = defaultdict(list)
        self.ent_pos_dict = defaultdict(list)
        self.ent_probas_dict = defaultdict(list)
        self.ents_pos_probas_dict = defaultdict(list)
        cnt = 0
        for n, (tok, tag, probas) in enumerate(zip(tokens, tags, tag_probas)):
            if tag.split('-')[-1] in self.entity_tags:
                f_tag = tag.split("-")[-1]
                if tag.startswith("B-") and any(self.ent_dict.values()):
                    for c_tag, entity in self.ent_dict.items():
                        self.add_entity(entity, c_tag)
                self.ent_dict[f_tag].append(tok)
                self.ent_pos_dict[f_tag].append(cnt)
                self.ent_probas_dict[f_tag].append(probas[self.tags_ind[tag]])

            elif any(self.ent_dict.values()):
                for tag, entity in self.ent_dict.items():
                    c_tag = tag.split("-")[-1]
                    self.add_entity(entity, c_tag)
            cnt += 1
        if any(self.ent_dict.values()):
            for tag, entity in self.ent_dict.items():
                c_tag = tag.split("-")[-1]
                self.add_entity(entity, c_tag)

        self.ents_pos_probas_dict = {tag: elements for tag, elements in self.ents_pos_probas_dict.items()
                                     if tag not in self.ignored_tags}

        for tag in self.ents_pos_probas_dict:
            ents_pos_proba = self.ents_pos_probas_dict[tag]

        entities_dict = {tag: [ent[0] for ent in ents] for tag, ents in self.ents_pos_probas_dict.items()}
        entities_positions_dict = {tag: [ent[1] for ent in ents] for tag, ents in self.ents_pos_probas_dict.items()}
        entities_probas_dict = {tag: [ent[2] for ent in ents] for tag, ents in self.ents_pos_probas_dict.items()}
        log.debug(f"entities_dict {entities_dict}")

        return entities_dict, entities_positions_dict, entities_probas_dict
