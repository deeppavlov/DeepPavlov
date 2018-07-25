"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import itertools
import json
import re
from collections import defaultdict

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("knowledge_base")
class KnowledgeBase(Estimator):

    def __init__(self, save_path, load_path=None, tokenizer=None, *args, **kwargs):
        super().__init__(save_path=save_path,
                         load_path=load_path,
                         *args, **kwargs)
        self.tokenizer = tokenizer
        self.kb = defaultdict(lambda: [])
        if self.load_path and self.load_path.is_file():
            self.load()

    def fit(self, *args):
        self.reset()
        self._update(*args)

    def _update(self, keys, kb_columns_list, kb_items_list):
        for key, cols, items in zip(keys, kb_columns_list, kb_items_list):
            if (None not in (key, items, cols)) and (key not in self.kb):
                kv_entry_list = (self._key_value_entries(item, cols)\
                                 for item in items)
                self.kb[key] = list(itertools.chain(*kv_entry_list))

    def _key_value_entries(self, kb_item, kb_columns):
        first_key = re.sub('\s+', '_', kb_item[kb_columns[0]].lower().strip())
        for col in kb_columns[1:]:
            if col in kb_item:
                second_key = re.sub('\s+', '_', col.lower().strip())
                key = first_key + '_' + second_key
                if self.tokenizer is not None:
                    yield (key, self.tokenizer([kb_item[col]])[0])
                else:
                    yield (key, kb_item[col])

    def __call__(self, keys):
# TODO: check if during validation kv is updated
        return [self.kb[key] for key in keys]

    def __len__(self):
        return len(self.kb)

    def keys(self):
        return self.kb.keys()

    def reset(self):
        self.kb = defaultdict(lambda: [])

    def save(self):
        log.info("[saving knowledge base to {}]".format(self.save_path))
        with self.save_path.open('wt', encoding='utf8') as fp:
            json.dump(self.kb, fp)

    def load(self):
        log.info("[loading knowledge base from {}]".format(self.load_path))
        with self.load_path.open('rt', encoding='utf8') as fp:
            self.kb.update(json.load(fp))


@register("knowledge_base_entity_normalizer")
class KnowledgeBaseEntityNormalizer(Component):

    def __init__(self, kb, denormalize=False, *args, **kwargs):
        self.kb = kb
        self.denormalize_flag = denormalize

    def normalize(self, key, tokens):
        utter = ' '.join(tokens)
        for entity, value in self.kb([key])[0]:
            to_replace = ' '.join(value)
            if to_replace:
                utter = utter.replace(to_replace, entity)
            else:
                log.debug("Empty value for knowledge base entry with key = {}"\
                          .format(key))
        return utter.split()

    def denormalize(self, key, tokens):
        for entity, value in self.kb([key])[0]:
            if entity in tokens:
                entity_pos = tokens.index(entity)
                tokens = tokens[:entity_pos] + value + tokens[entity_pos + 1:] 
        return tokens

    def __call__(self, keys, values, kb_columns_list=None, kb_items_list=None):
        if None not in (kb_columns_list, kb_items_list):
            self.kb._update(keys, kb_columns_list, kb_items_list)
        if self.denormalize_flag:
            return [self.denormalize(key, val) for key, val in zip(keys, values)]
        return [self.normalize(key, val) for key, val in zip(keys, values)]

