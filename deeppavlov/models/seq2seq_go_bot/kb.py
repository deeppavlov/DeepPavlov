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
import json
import re
from typing import Callable, List, Dict
from collections import defaultdict

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("knowledge_base")
class KnowledgeBase(Estimator):
    """
    A custom dictionary that encodes knowledge facts from :class:`~deeppavlov.dataset_readers.kvret_reader.KvretDatasetReader` data.

    Example:
        .. code:: python

            >>> from models.seq2seq_go_bot.kb import KnowledgeBase
            >>> kb = KnowledgeBase(save_path="kb.json", load_path="kb.json")
            >>> kb.fit(['person1'], [['name', 'hair', 'eyes']], [[{'name': 'Sasha', 'hair': 'long dark', 'eyes': 'light blue '}]])

            >>> kb(['person1'])
            [[('sasha_hair', 'long dark'), ('sasha_eyes', 'light blue ')]]

            >>> kb(['person_that_doesnt_exist'])
            [[]]

    Parameters:
        save_path: path to save the dictionary with knowledge.
        load_path: path to load the json with knowledge.
        tokenizer: tokenizer used to split entity values into tokens.
        **kwargs: parameters passed to parent :class:`~deeppavlov.core.models.estimator.Estimator`.
    """
    def __init__(self,
                 save_path: str,
                 load_path: str = None,
                 tokenizer: Callable = None,
                 *args, **kwargs) -> None:
        super().__init__(save_path=save_path,
                         load_path=load_path,
                         *args, **kwargs)
        self.tokenizer = tokenizer
        self.kb = defaultdict(lambda: [])
        self.primary_keys = []
        if self.load_path and self.load_path.is_file():
            self.load()

    def fit(self, *args):
        self.reset()
        self._update(*args)

    def _update(self, keys, kb_columns_list, kb_items_list, update_primary_keys=True):
        for key, cols, items in zip(keys, kb_columns_list, kb_items_list):
            if (None not in (key, items, cols)) and (key not in self.kb):
                kv_entry_list = (self._key_value_entries(item, cols,
                                                         update=update_primary_keys)
                                 for item in items)
                self.kb[key] = list(itertools.chain(*kv_entry_list))

    def _key_value_entries(self, kb_item, kb_columns, update=True):
        def _format(s):
            return re.sub('\s+', '_', s.lower().strip())
        first_key = _format(kb_item[kb_columns[0]])
        for col in kb_columns:
            key = first_key + '_' + _format(col)
            if update and (key not in self.primary_keys):
                self.primary_keys.append(key)
            if col in kb_item:
                if self.tokenizer is not None:
                    yield (key, self.tokenizer([kb_item[col]])[0])
                else:
                    yield (key, kb_item[col])

    def __call__(self, keys, kb_columns_list=None, kb_items_list=None):
        if None not in (kb_columns_list, kb_items_list):
            self._update(keys, kb_columns_list, kb_items_list, update_primary_keys=False)
        res = []
        for key in keys:
            res.append(self.kb[key])
            for k, value in res[-1]:
                if k not in self.primary_keys:
                    raise ValueError("Primary key `{}` is not present in knowledge base"\
                                     .format(k))
        return res

    def __len__(self):
        return len(self.kb)

    def keys(self):
        return self.kb.keys()

    def reset(self):
        self.kb = defaultdict(lambda: [])
        self.primary_keys = []

    def save(self):
        log.info("[saving knowledge base to {}]".format(self.save_path))
        json.dump(self.kb, self.save_path.open('wt'))
        json.dump(self.primary_keys, self.save_path.with_suffix('.keys.json').open('wt'))

    def load(self):
        log.info("[loading knowledge base from {}]".format(self.load_path))
        self.kb.update(json.load(self.load_path.open('rt')), primary_keys=False)
        self.primary_keys = json.load(self.load_path.with_suffix('.keys.json').open('rt'))
=======
        with self.save_path.open('wt', encoding='utf8') as fp:
            json.dump(self.kb, fp)

    def load(self):
        log.info("[loading knowledge base from {}]".format(self.load_path))
        with self.load_path.open('rt', encoding='utf8') as fp:
            self.kb.update(json.load(fp))
>>>>>>> origin/dev


@register("knowledge_base_entity_normalizer")
class KnowledgeBaseEntityNormalizer(Component):
    """
    Uses instance of :class:`~deeppavlov.models.seq2seq_go_bot.kb.KnowledgeBase` to normalize or to undo normalization of entities in the input utterance.

    To normalize is to substitute all mentions of database entities with their normalized form.

    To undo normalization is to substitute all mentions of database normalized entities with their original form.

    Example:
        .. code:: python

            >>> from models.seq2seq_go_bot.kb import KnowledgeBase
            >>> kb = KnowledgeBase(save_path="kb.json", load_path="kb.json")
            >>> kb.fit(['person1'], [['name', 'hair', 'eyes']], [[{'name': 'Sasha', 'hair': 'long dark', 'eyes': 'light blue '}]])
            >>> kb(['person1'])
            [[('sasha_hair', 'long dark'), ('sasha_eyes', 'light blue ')]]

            >>> from models.seq2seq_go_bot.kb import KnowledgeBaseEntityNormalizer
            >>> normalizer = KnowledgeBaseEntityNormalizer(kb=kb, denormalize=False)
            >>> normalizer(['person1'], [["some", "guy", "with", "long", "dark", "hair", "said", "hi"]])
            [['some', 'guy', 'with', 'sasha_hair', 'hair', 'said', 'hi']]

            >>> denormalizer = KnowledgeBaseEntityNormalizer(kb=kb, denormalize=True)
            >>> denormalizer(['person1'], [['some', 'guy', 'with', 'sasha_hair', 'hair', 'said', 'hi']])
            [['some', 'guy', 'with', 'long', 'dark', 'hair', 'said', 'hi']]


    Parameters:
        kb: knowledge base of type :class:`~deeppavlov.models.seq2seq_go_bot.KnowledgeBase`.
        denormalize: flag indicates whether to normalize or to undo normalization ("denormalize").
        **kwargs: parameters passed to parent :class:`~deeppavlov.core.models.component.Component` class.

    def __init__(self, remove=False, denormalize=False, *args, **kwargs):
        self.denormalize_flag = denormalize
        self.remove = remove

    def normalize(self, tokens, entries):
        for entity, value_tokens in sorted(entries, key=lambda e: -len(e[1])):
            value_len = len(value_tokens)
            if ' '.join(value_tokens).strip():
                for i in range(len(tokens)):
                    if tokens[i:i+value_len] == value_tokens:
                        if self.remove:
                            tokens = tokens[:i] + tokens[i+value_len:]
                        else:
                            tokens = tokens[:i] + [entity] + tokens[i+value_len:]
        return tokens

    def denormalize(self, tokens, entries):
        for entity, value in entries:
            while (entity in tokens):
                entity_pos = tokens.index(entity)
                tokens = tokens[:entity_pos] + value + tokens[entity_pos+1:] 
        return tokens

    def __call__(self, 
                 tokens_list: List[str],
                 entries_list) -> List[List[str]]:
        # TODO: entries_list type
        if self.denormalize_flag:
            return [self.denormalize(t, e) for t, e in zip(tokens_list, entries_list)]
        return [self.normalize(t, e) for t, e in zip(tokens_list, entries_list)]
