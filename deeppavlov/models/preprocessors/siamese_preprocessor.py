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

import numpy as np
from deeppavlov.core.commands.utils import expand_path
from keras.preprocessing.sequence import pad_sequences
from deeppavlov.core.common.log import get_logger
import random
import copy

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator
from typing import List, Callable
from deeppavlov.core.data.utils import mark_done, is_done, zero_pad_truncate

log = get_logger(__name__)


@register('siamese_preprocessor')
class SiamesePreprocessor(Estimator):
    """Class to encode characters, tokens, whole contexts and responses with vocabularies, to pad and truncate.

    Args:
        max_sequence_length: A maximum length of a sequence in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        max_token_length: A maximum length of a token for representing it by a character-level embedding.
        padding: Padding. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a sequence will be padded at the beginning.
            If set to ``post`` it will padded at the end.
        truncating: Truncating. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a sequence will be truncated at the beginning.
            If set to ``post`` it will truncated at the end.
        token_embeddings: Whether to use token embeddins or not.
        char_embeddings: Whether to use character embeddings or not.
        char_pad: Character-level padding. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a token will be padded at the beginning.
            If set to ``post`` it will padded at the end.
        char_trunc: Character-level truncating. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a token will be truncated at the beginning.
            If set to ``post`` it will truncated at the end.
        tok_dynamic_batch:  Whether to use dynamic batching. If ``True``, a maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        char_dynamic_batch: Whether to use dynamic batching for character-level embeddings.
            If ``True``, a maximum length of a token for a batch
            will be equal to the maximum of all tokens lengths from this batch,
            but not higher than ``max_token_length``.
        update_embeddings: Whether to store and update context and response embeddings or not.
        pos_pool_sample: Whether to sample response from `pos_pool` each time when the batch is generated.
            If ``False``, the response from `response` will be used.
        pos_pool_rank: Whether to count samples from the whole `pos_pool` as correct answers in test / validation mode.
        tokenizer: The method to tokenize contexts and responses.
        seed: Random seed.
        embedder: The method providing embeddings for tokens.
        embedding_dim: Dimensionality of token (word) embeddings.
        use_matrix: Whether to use trainable matrix with token (word) embeddings.
    """

    def __init__(self,
                 save_path: str,
                 load_path: str,
                 max_sequence_length: int,
                 use_matrix: bool,
                 embedder: Component,
                 max_token_length: int = None,
                 padding: str = 'post',
                 truncating: str = 'post',
                 token_embeddings: bool = True,
                 char_embeddings: bool = False,
                 char_pad: str = 'post',
                 char_trunc: str = 'post',
                 tok_dynamic_batch: bool = False,
                 char_dynamic_batch: bool = False,
                 update_embeddings: bool = False,
                 num_ranking_samples: int = 10,
                 num_context_turns: int = 1,
                 tokenizer: Callable = None,
                 vocab: Callable = "dialog_vocab",
                 embedding_dim: int = 300,
                 **kwargs):

        self.max_sequence_length = max_sequence_length
        self.token_embeddings = token_embeddings
        self.char_embeddings = char_embeddings
        self.max_token_length = max_token_length
        self.padding = padding
        self.truncating = truncating
        self.char_pad = char_pad
        self.char_trunc = char_trunc
        self.tok_dynamic_batch = tok_dynamic_batch
        self.char_dynamic_batch = char_dynamic_batch
        self.update_embeddings = update_embeddings
        self.num_ranking_samples = num_ranking_samples
        self.num_context_turns = num_context_turns
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.use_matrix = use_matrix

        self.save_path = expand_path(save_path).resolve()
        self.load_path = expand_path(load_path).resolve()

        super().__init__(load_path=self.load_path, save_path=self.save_path, **kwargs)

        self.len_vocab = 0
        self.len_char_vocab = 0
        self.emb_matrix = None

    def destroy(self):
        self.embedder.destroy()

    def fit(self, x):
            x_tok = [self.tokenizer(el) for el in x]
            self.vocab.fit([el for x in x_tok for el in x])

    def __call__(self, x):
        x_cut = [el[:self.num_context_turns+self.num_ranking_samples] for el in x]
        for el in x_cut:
            x_tok = self.tokenizer(el)
            x_ctok = [y if len(y) != 0 else [''] for y in x_tok]
            if self.use_matrix:
                x_proc = self.vocab(x_ctok)
            else:
                x_proc = self.embedder(x_ctok)
            x_proc = zero_pad_truncate(x_proc, self.max_sequence_length)
            x_proc = list(x_proc)
            yield x_proc


    def load(self):
        pass
        # self.vocab.load()

    def save(self):
        self.vocab.save()
