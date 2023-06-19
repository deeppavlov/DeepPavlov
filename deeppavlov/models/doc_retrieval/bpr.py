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

from typing import Tuple

import faiss
import numpy as np
import torch
from tqdm import trange
from transformers import AutoTokenizer, BertModel

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable


class FaissBinaryIndex:
    def __init__(self, index: faiss.Index):
        self.index = index

    def search(self, query_embs: np.ndarray, k: int, binary_k=1000, rerank=True) -> Tuple[np.ndarray, np.ndarray]:
        faiss.omp_set_num_threads(12)
        num_queries = query_embs.shape[0]
        bin_query_embs = np.packbits(np.where(query_embs > 0, 1, 0)).reshape(num_queries, -1)

        raw_index = self.index.index
        _, ids_arr = raw_index.search(bin_query_embs, binary_k)
        psg_embs = np.vstack([np.unpackbits(raw_index.reconstruct(int(id_))) for id_ in ids_arr.reshape(-1)])
        psg_embs = psg_embs.reshape(query_embs.shape[0], binary_k, query_embs.shape[1])
        psg_embs = psg_embs.astype(np.float32)

        psg_embs = psg_embs * 2 - 1
        scores_arr = np.einsum("ijk,ik->ij", psg_embs, query_embs)
        sorted_indices = np.argsort(-scores_arr, axis=1)

        ids_arr = ids_arr[np.arange(num_queries)[:, None], sorted_indices]
        ids_arr = np.array([self.index.id_map.at(int(id_)) for id_ in ids_arr.reshape(-1)], dtype=np.int)
        ids_arr = ids_arr.reshape(num_queries, -1)
        scores_arr = scores_arr[np.arange(num_queries)[:, None], sorted_indices]

        return scores_arr[:, :k], ids_arr[:, :k]


@register('bpr')
class BPR(Component, Serializable):
    def __init__(self, pretrained_model: str,
                 load_path: str,
                 bpr_index: str,
                 query_encoder_file: str,
                 max_query_length: int = 256,
                 top_n: int = 100,
                 device: str = "gpu",
                 *args, **kwargs
                 ):
        super().__init__(save_path=None, load_path=load_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.bpr_index = bpr_index
        self.top_n = top_n
        self.max_query_length = max_query_length
        self.query_encoder_file = query_encoder_file
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.q_encoder = BertModel.from_pretrained(pretrained_model).to(self.device)
        self.load()
        self.index = FaissBinaryIndex(self.base_index)

    def load(self):
        checkpoint = torch.load(str(self.load_path / self.query_encoder_file), map_location=self.device)
        self.q_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.base_index = faiss.read_index_binary(str(self.load_path / self.bpr_index))

    def save(self) -> None:
        pass

    def encode_queries(self, queries, batch_size: int = 256) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for start in trange(0, len(queries), batch_size):
                model_inputs = self.tokenizer.batch_encode_plus(
                    queries[start: start + batch_size],
                    return_tensors="pt",
                    max_length=self.max_query_length,
                    padding="max_length",
                )
                model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
                sequence_output = self.q_encoder(**model_inputs)[0]
                emb = sequence_output[:, 0, :].contiguous().cpu().numpy()
                embeddings.append(emb)

        return np.vstack(embeddings)

    def __call__(self, queries):
        queries = [query.lower() for query in queries]
        query_embeddings = self.encode_queries(queries)
        scores_batch, ids_batch = self.index.search(query_embeddings, self.top_n)
        ids_batch = ids_batch.tolist()
        return ids_batch
