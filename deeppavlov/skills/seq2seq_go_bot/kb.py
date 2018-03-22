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
from collections import defaultdict

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register("knowledge_base")
class KnowledgeBase(Estimator):

    def __init__(self, save_path, load_path, *args, **kwargs):
        super().__init__(save_path=save_path,
                         load_path=load_path,
                         *args, **kwargs)
        self.kb = defaultdict(lambda: {})

    def fit(self, *args):
        self.reset()
        for key, kb_columns, kb_items in zip(*args):
            if None not in (kb_items, kb_columns):
                kv_entry_list = (self._key_value_entries(item, kb_columns)\
                                 for item in kb_items)
                self.kb[key] = list(itertools.chain(*kv_entry_list))

    @staticmethod
    def _key_value_entries(kb_item, kb_columns):
        first_key = kb_item[kb_columns[0]]
        for col in kb_columns[1:]:
            if col in kb_item:
                yield (first_key, col, kb_item[col])

    def __call__(self, batch):
        return [self.kb[dialog_id] for dialog_id in batch]

    def __len__(self):
        return len(self.kb)

    def keys(self):
        return self.kb.keys()

    def reset(self):
        self.kb = defaultdict(lambda: {})

    def save(self):
        log.info("[saving knowledge base to {}]".format(self.save_path))
        json.dump(self.kb, self.save_path.open('wt'))

    def load(self):
        log.info("[loading knowledge base from {}]".format(self.load_path))
        self.kb = json.load(self.load_path.open('rt'))


@register("knowledge_base_entity_normalizer")
class KnowledgeBaseEntityNormalizer(Component):

    def __init__(self, kb, denormalize=False, *args, **kwargs):
        self.kb = kb
        self.denormalize = denormalize

    def __call__(self, batch):
# TODO normalization and denormalization
        return batch
