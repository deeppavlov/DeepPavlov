# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
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
from typing import List

import torch
import transformers

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.serializable import Serializable


def _filter(data_batch: List[List[float]], mask_batch: List[List[int]]):
    return [[tok_data for tok_data, tok_mask in zip(sent_data, sent_mask) if tok_mask]
            for sent_data, sent_mask in zip(data_batch, mask_batch)]


@register('transformers_embedder')
class TransformersEmbedder(Serializable):
    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self):
        self.model = transformers.BertModel.from_pretrained(self.load_path, config=self.config).eval().to(self.device)

    def __init__(self, load_path, bert_config_path=None, **kwargs):
        super().__init__(save_path=None, load_path=load_path, **kwargs)
        if bert_config_path is not None:
            bert_config_path = expand_path(bert_config_path)
        self.config = bert_config_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load()

    def __call__(self, subtoken_ids_batch, startofwords_batch, attention_batch):
        ids_tensor = torch.tensor(subtoken_ids_batch).to(self.device)
        attention_tensor = torch.tensor(attention_batch).to(self.device)
        with torch.no_grad():
            outputs = self.model(ids_tensor, attention_tensor)
        outputs = outputs[0].cpu().numpy()
        word_emb = _filter(outputs, startofwords_batch)
        subword_emb = _filter(outputs, attention_batch)
        return word_emb, subword_emb
