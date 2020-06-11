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
from pathlib import Path
from typing import Tuple, Iterator, Optional, Dict, List, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.core.data.utils import chunk_generator
from deeppavlov.dataset_iterators.file_paths_iterator import FilePathsIterator
from deeppavlov.models.preprocessors.str_utf8_encoder import StrUTF8Encoder

log = getLogger(__name__)


@register('elmo_file_paths_iterator')
class ELMoFilePathsIterator(FilePathsIterator):
    """Dataset iterator for tokenized datasets like 1 Billion Word Benchmark
    It gets lists of file paths from the data dictionary and returns batches of lines from each file.

    Args:
        data: dict with keys ``'train'``, ``'valid'`` and ``'test'`` and values
        load_path: path to the vocabulary to be load from
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching
        unroll_steps: number of unrolling steps
        n_gpus: number of gpu to use
        max_word_length: max length of word
        bos: tag of begin of sentence
        eos: tag of end of sentence

    """

    def __init__(self,
                 data: Dict[str, List[Union[str, Path]]],
                 load_path: Union[str, Path],
                 seed: Optional[int] = None,
                 shuffle: bool = True,
                 unroll_steps: Optional[int] = None,
                 n_gpus: Optional[int] = None,
                 max_word_length: Optional[int] = None,
                 bos: str = "<S>",
                 eos: str = "</S>",
                 *args, **kwargs) -> None:
        self.unroll_steps = unroll_steps
        self.n_gpus = n_gpus
        self.bos = bos
        self.eos = eos
        self.str_utf8_encoder = StrUTF8Encoder(
            max_word_length=max_word_length,
            pad_special_char_use=True,
            word_boundary_special_char_use=True,
            sentence_boundary_special_char_use=False,
            reversed_sentense_tokens=False,
            bos=self.bos,
            eos=self.eos,
            save_path=load_path,
            load_path=load_path,
        )
        self.simple_vocab = SimpleVocabulary(
            min_freq=2,
            special_tokens=[self.eos, self.bos, "<UNK>"],
            unk_token="<UNK>",
            freq_drop_load=True,
            save_path=load_path,
            load_path=load_path,
        )
        super().__init__(data, seed, shuffle, *args, **kwargs)

    def _line2ids(self, line):
        line = [self.bos] + line.split() + [self.eos]

        char_ids = self.str_utf8_encoder(line)
        reversed_char_ids = list(reversed(char_ids))
        char_ids = char_ids[:-1]
        reversed_char_ids = reversed_char_ids[:-1]

        token_ids = self.simple_vocab(line)
        reversed_token_ids = list(reversed(token_ids))
        token_ids = token_ids[1:]
        reversed_token_ids = reversed_token_ids[1:]

        return char_ids, reversed_char_ids, token_ids, reversed_token_ids

    def _line_generator(self, shard_generator):
        for shard in shard_generator:
            line_generator = chunk_generator(shard, 1)
            for line in line_generator:
                line = line[0]
                char_ids, reversed_char_ids, token_ids, reversed_token_ids = \
                    self._line2ids(line)
                yield char_ids, reversed_char_ids, token_ids, reversed_token_ids

    @staticmethod
    def _batch_generator(line_generator, batch_size, unroll_steps):
        batch = [[[] for i in range(4)] for i in range(batch_size)]
        stream = [[[] for i in range(4)] for i in range(batch_size)]

        try:
            while True:
                for batch_item, stream_item in zip(batch, stream):
                    while len(stream_item[0]) < unroll_steps:
                        line = next(line_generator)
                        for sti, lni in zip(stream_item, line):
                            sti.extend(lni)
                    for sti, bchi in zip(stream_item, batch_item):
                        _b = sti[:unroll_steps]
                        _s = sti[unroll_steps:]
                        bchi.clear()
                        _b = _b
                        bchi.extend(_b)

                        sti.clear()
                        sti.extend(_s)
                char_ids, reversed_char_ids, token_ids, reversed_token_ids = \
                    zip(*batch)
                yield char_ids, reversed_char_ids, token_ids, reversed_token_ids
        except StopIteration:
            pass

    def gen_batches(self, batch_size: int, data_type: str = 'train', shuffle: Optional[bool] = None) \
            -> Iterator[Tuple[str, str]]:
        if shuffle is None:
            shuffle = self.shuffle

        tgt_data = self.data[data_type]
        shard_generator = self._shard_generator(tgt_data, shuffle=shuffle)
        line_generator = self._line_generator(shard_generator)

        if data_type == 'train':
            unroll_steps = self.unroll_steps
            n_gpus = self.n_gpus
        else:
            unroll_steps = 1
            batch_size = 256
            n_gpus = 1

        batch_generator = self._batch_generator(line_generator, batch_size * n_gpus, unroll_steps)

        for char_ids, reversed_char_ids, token_ids, reversed_token_ids in batch_generator:
            batch = [(char_ids, reversed_char_ids), (token_ids, reversed_token_ids)]
            yield batch
