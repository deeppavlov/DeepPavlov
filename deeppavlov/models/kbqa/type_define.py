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
from typing import List

import pymorphy2
import spacy
from nltk.corpus import stopwords

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register


@register('answer_types_extractor')
class AnswerTypesExtractor:
    """Class which defines answer types for the question"""

    def __init__(self, lang: str, types_filename: str, types_sets_filename: str,
                 num_types_to_return: int = 15, **kwargs):
        """

        Args:
            lang: Russian or English
            types_filename: filename with dictionary where keys are type ids and values are type labels
            types_sets_filename: filename with dictionary where keys are NER tags and values are Wikidata types
                corresponding to tags
            num_types_to_return: how many answer types to return for each question
            **kwargs:
        """
        self.lang = lang
        self.types_filename = str(expand_path(types_filename))
        self.types_sets_filename = str(expand_path(types_sets_filename))
        self.num_types_to_return = num_types_to_return
        self.morph = pymorphy2.MorphAnalyzer()
        if self.lang == "@en":
            self.stopwords = set(stopwords.words("english"))
            self.nlp = spacy.load("en_core_web_sm")
            self.pronouns = ["what"]
        elif self.lang == "@ru":
            self.stopwords = set(stopwords.words("russian"))
            self.nlp = spacy.load("ru_core_news_sm")
            self.pronouns = ["какой", "каком"]
        with open(self.types_filename, 'rb') as fl:
            self.types_dict = pickle.load(fl)
        with open(self.types_sets_filename, 'rb') as fl:
            self.types_sets = pickle.load(fl)

    def __call__(self, questions_batch: List[str], entity_substr_batch: List[List[str]],
                 tags_batch: List[List[str]], types_substr_batch: List[List[str]] = None):
        types_sets_batch = []
        if types_substr_batch is None:
            types_substr_batch = []
            for question, entity_substr_list in zip(questions_batch, entity_substr_batch):
                types_substr = []
                type_noun = ""
                doc = self.nlp(question)
                token_pos_dict = {}
                for n, token in enumerate(doc):
                    token_pos_dict[token.text] = n
                for token in doc:
                    if token.text.lower() in self.pronouns and token.head.dep_ in ["attr", "nsubj"]:
                        type_noun = token.head.text
                        if not any([type_noun in entity_substr.lower() for entity_substr in entity_substr_list]):
                            types_substr.append(type_noun)
                        break
                if type_noun:
                    for token in doc:
                        if token.head.text == type_noun and token.dep_ in ["amod", "compound"]:
                            type_adj = token.text
                            if not any([type_adj.lower() in entity_substr.lower() for entity_substr in
                                        entity_substr_list]):
                                types_substr.append(type_adj)
                            break
                        elif token.head.text == type_noun and token.dep_ == "prep":
                            if len(list(token.children)) == 1 \
                                    and not any([[tok.text for tok in token.children][0] in entity_substr.lower()
                                                 for entity_substr in entity_substr_list]):
                                types_substr += [token.text, [tok.text for tok in token.children][0]]
                elif any([word in question for word in self.pronouns]):
                    for token in doc:
                        if token.dep_ == "nsubj" and not any([token.text in entity_substr.lower()
                                                              for entity_substr in entity_substr_list]):
                            types_substr.append(token.text)

                types_substr = [(token, token_pos_dict[token]) for token in types_substr]
                types_substr = sorted(types_substr, key=lambda x: x[1])
                types_substr = " ".join([elem[0] for elem in types_substr])
                types_substr_batch.append(types_substr)
        for types_substr in types_substr_batch:
            types_substr_tokens = types_substr.split()
            types_substr_tokens = [tok for tok in types_substr_tokens if tok not in self.stopwords]
            if self.lang == "@ru":
                types_substr_tokens = [self.morph.parse(tok)[0].normal_form for tok in types_substr_tokens]
            types_substr_tokens = set(types_substr_tokens)
            types_scores = []
            for entity in self.types_dict:
                labels, cnt = self.types_dict[entity]
                cur_cnts = []
                for label in labels:
                    label_tokens = label.lower().split()
                    if len(types_substr_tokens) == 1 and len(label_tokens) == 2 and \
                            list(types_substr_tokens)[0] == label_tokens[0]:
                        cur_cnts.append(0.3)
                    else:
                        inters = types_substr_tokens.intersection(set(label_tokens))
                        cur_cnts.append(len(inters) / max(len(types_substr_tokens), len(label_tokens)))

                types_scores.append([entity, max(cur_cnts), cnt])
            types_scores = sorted(types_scores, key=lambda x: (x[1], x[2]), reverse=True)
            cur_types = [elem[0] for elem in types_scores if elem[1] > 0][:self.num_types_to_return]
            types_sets_batch.append(cur_types)

        for n, (question, types_sets) in enumerate(zip(questions_batch, types_sets_batch)):
            question = question.lower()
            if not types_sets:
                if self.lang == "@ru":
                    if question.startswith("кто"):
                        types_sets_batch[n] = self.types_sets["PER"]
                    elif question.startswith("где"):
                        types_sets_batch[n] = self.types_sets["LOC"]
                elif self.lang == "@en":
                    if question.startswith("who"):
                        types_sets_batch[n] = self.types_sets["PER"]
                    elif question.startswith("where"):
                        types_sets_batch[n] = self.types_sets["LOC"]

        new_entity_substr_batch, new_entity_offsets_batch, new_tags_batch = [], [], []
        for question, entity_substr_list, tags_list in zip(questions_batch, entity_substr_batch, tags_batch):
            new_entity_substr, new_tags = [], []
            if not entity_substr_list:
                doc = self.nlp(question)
                for token in doc:
                    if token.dep_ == "nsubj":
                        new_entity_substr.append(token.text)
                        new_tags.append("MISC")
                        break
                new_entity_substr_batch.append(new_entity_substr)
                new_tags_batch.append(new_tags)
            else:
                new_entity_substr_batch.append(entity_substr_list)
                new_tags_batch.append(tags_list)

        return types_sets_batch, new_entity_substr_batch, new_tags_batch
