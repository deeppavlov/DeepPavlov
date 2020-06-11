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

from logging import getLogger
from typing import List, Union, Iterable, Optional

import numpy as np

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad_truncate
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

log = getLogger(__name__)


@register('siamese_preprocessor')
class SiamesePreprocessor(Estimator):
    """ Preprocessing of data samples containing text strings to feed them in a siamese network.

    First ``num_context_turns`` strings in each data sample corresponds to the dialogue ``context``
    and the rest string(s) in the sample is (are) ``response(s)``.

    Args:
        save_path: The parameter is only needed to initialize the base class
            :class:`~deeppavlov.core.models.serializable.Serializable`.
        load_path: The parameter is only needed to initialize the base class
            :class:`~deeppavlov.core.models.serializable.Serializable`.
        max_sequence_length: A maximum length of text sequences in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        dynamic_batch:  Whether to use dynamic batching. If ``True``, the maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        padding: Padding. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a sequence will be padded at the beginning.
            If set to ``post`` it will padded at the end.
        truncating: Truncating. Possible values are ``pre`` and ``post``.
            If set to ``pre`` a sequence will be truncated at the beginning.
            If set to ``post`` it will truncated at the end.
        use_matrix: Whether to use a trainable matrix with token (word) embeddings.
        num_context_turns: A number of ``context`` turns in data samples.
        num_ranking_samples: A number of condidates for ranking including positive one.
        add_raw_text: whether add raw text sentences to output data list or not.
            Use with conjunction of models using sentence encoders
        tokenizer: An instance of one of the :class:`deeppavlov.models.tokenizers`.
        vocab: An instance of :class:`deeppavlov.core.data.simple_vocab.SimpleVocabulary`.
        embedder: an instance of one of the :class:`deeppavlov.models.embedders`.
        sent_vocab: An instance of of :class:`deeppavlov.core.data.simple_vocab.SimpleVocabulary`.
            It is used to store all ``responces`` and to find the best ``response``
            to the user ``context`` in the ``interact`` mode.
    """

    def __init__(self,
                 save_path: str = './tok.dict',
                 load_path: str = './tok.dict',
                 max_sequence_length: int = None,
                 dynamic_batch: bool = False,
                 padding: str = 'post',
                 truncating: str = 'post',
                 use_matrix: bool = True,
                 num_context_turns: int = 1,
                 num_ranking_samples: int = 1,
                 add_raw_text: bool = False,
                 tokenizer: Component = None,
                 vocab: Optional[Estimator] = None,
                 embedder: Optional[Component] = None,
                 sent_vocab: Optional[Estimator] = None,
                 **kwargs):

        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.truncating = truncating
        self.dynamic_batch = dynamic_batch
        self.use_matrix = use_matrix
        self.num_ranking_samples = num_ranking_samples
        self.num_context_turns = num_context_turns
        self.add_raw_text = add_raw_text
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.vocab = vocab
        self.sent_vocab = sent_vocab
        self.save_path = expand_path(save_path).resolve()
        self.load_path = expand_path(load_path).resolve()

        super().__init__(load_path=self.load_path, save_path=self.save_path, **kwargs)

    def fit(self, x: List[List[str]]) -> None:
        if self.sent_vocab is not None:
            self.sent_vocab.fit([el[self.num_context_turns:] for el in x])
        x_tok = [self.tokenizer(el) for el in x]
        self.vocab.fit([el for x in x_tok for el in x])

    def __call__(self, x: Union[List[List[str]], List[str]]) -> Iterable[List[List[np.ndarray]]]:
        if len(x) == 0 or isinstance(x[0], str):
            if len(x) == 1:  # interact mode: len(batch) == 1
                x_preproc = [[sent.strip() for sent in x[0].split('&')]]  # List[str] -> List[List[str]]
            elif len(x) == 0:
                x_preproc = [['']]
            else:
                x_preproc = [[el] for el in x]
        else:
            x_preproc = [el[:self.num_context_turns + self.num_ranking_samples] for el in x]
        for el in x_preproc:
            x_tok = self.tokenizer(el)
            x_ctok = [y if len(y) != 0 else [''] for y in x_tok]
            if self.use_matrix:
                x_proc = self.vocab(x_ctok)
            else:
                x_proc = self.embedder(x_ctok)
            if self.dynamic_batch:
                msl = min((max([len(y) for el in x_tok for y in el]), self.max_sequence_length))
            else:
                msl = self.max_sequence_length
            x_proc = zero_pad_truncate(x_proc, msl, pad=self.padding, trunc=self.truncating)
            x_proc = list(x_proc)
            if self.add_raw_text:
                x_proc += el  # add (self.num_context_turns+self.num_ranking_samples) raw sentences
            yield x_proc

    def load(self) -> None:
        pass

    def save(self) -> None:
        if self.sent_vocab is not None:
            self.sent_vocab.save()
        self.vocab.save()
