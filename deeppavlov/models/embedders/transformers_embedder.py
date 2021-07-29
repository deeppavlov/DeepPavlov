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
from pathlib import Path
from typing import Union, Tuple, Collection

import torch
import transformers

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.serializable import Serializable


@register('transformers_bert_embedder')
class TransformersBertEmbedder(Serializable):
    """Transformers-based BERT model for embeddings tokens, subtokens and sentences

    Args:
        load_path: path to a pretrained BERT pytorch checkpoint
        bert_config_file: path to a BERT configuration file
        truncate: whether to remove zero-paddings from returned data

    """
    model: transformers.BertModel
    dim: int

    def __init__(self, load_path: Union[str, Path], bert_config_path: Union[str, Path] = None,
                 truncate: bool = False, **kwargs):
        super().__init__(save_path=None, load_path=load_path, **kwargs)
        if bert_config_path is not None:
            bert_config_path = expand_path(bert_config_path)
        self.config = bert_config_path
        self.truncate = truncate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load()

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self):
        self.model = transformers.BertModel.from_pretrained(self.load_path, config=self.config).eval().to(self.device)
        self.dim = self.model.config.hidden_size

    def __call__(self,
                 subtoken_ids_batch: Collection[Collection[int]],
                 startofwords_batch: Collection[Collection[int]],
                 attention_batch: Collection[Collection[int]]) -> Tuple[Collection[Collection[Collection[float]]],
                                                                        Collection[Collection[Collection[float]]],
                                                                        Collection[Collection[float]],
                                                                        Collection[Collection[float]],
                                                                        Collection[Collection[float]]]:
        """Predict embeddings values for a given batch

        Args:
            subtoken_ids_batch: padded indexes for every subtoken
            startofwords_batch: a mask matrix with ``1`` for every first subtoken init in a token and ``0``
                for every other subtoken
            attention_batch: a mask matrix with ``1`` for every significant subtoken and ``0`` for paddings
        """
        ids_tensor = torch.tensor(subtoken_ids_batch, device=self.device, dtype=torch.long)
        startofwords_tensor = torch.tensor(startofwords_batch, device=self.device).bool()
        attention_tensor = torch.tensor(attention_batch, device=self.device)
        with torch.no_grad():
            output = self.model(ids_tensor, attention_tensor)
            last_hidden = output.last_hidden_state
            pooler_output = output.pooler_output
            attention_tensor = attention_tensor.unsqueeze(-1)
            max_emb = torch.max(last_hidden - 1e9 * (1 - attention_tensor), dim=1)[0]
            subword_emb = last_hidden * attention_tensor
            mean_emb = torch.sum(subword_emb, dim=1) / torch.sum(attention_tensor, dim=1)

            tokens_lengths = startofwords_tensor.sum(dim=1)
            word_emb = torch.zeros((subword_emb.shape[0], tokens_lengths.max(), subword_emb.shape[2]),
                                   device=self.device, dtype=subword_emb.dtype)
            target_indexes = (torch.arange(word_emb.shape[1], device=self.device).expand(word_emb.shape[:-1]) <
                              tokens_lengths.unsqueeze(-1))
            word_emb[target_indexes] = subword_emb[startofwords_tensor]

        subword_emb = subword_emb.cpu().numpy()
        word_emb = word_emb.cpu().numpy()
        pooler_output = pooler_output.cpu().numpy()
        max_emb = max_emb.cpu().numpy()
        mean_emb = mean_emb.cpu().numpy()
        if self.truncate:
            subword_emb = [item[:mask.sum()] for item, mask in zip(subword_emb, attention_batch)]
            word_emb = [item[:mask.sum()] for item, mask in zip(word_emb, startofwords_batch)]
        return word_emb, subword_emb, max_emb, mean_emb, pooler_output
