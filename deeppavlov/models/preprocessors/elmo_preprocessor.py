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
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
from nltk import word_tokenize
from tqdm import tqdm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

logger = get_logger(__name__)

@register('elmo_str_spec_tokens_wrapper')
class ELMoStrSpecialTokensWrapper(Component):
    """Component for wrapping strings in special tokens

    Args:
        bos: The parameter is only needed to add special token to beginning of strings.
        eos: The parameter is only needed to add special token to end of strings.
        reverse: The parameter is only needed to wrap reversed strings.
        tokenized: The parameter is only needed to wrap tokenized strings.
    """
    def __init__(self, 
                 bos:str = '',  
                 eos:str = '', 
                 reverse:bool = False,
                 tokenized:bool = False, 
                 *args, 
                 **kwargs):
        self.bos = bos
        self.eos = eos
        self.reverse = reverse
        self.tokenized = tokenized
    
    @staticmethod
    def _wrap(raw_tokens, reverse, bos, eos):
        if reverse:
            return [eos] + raw_tokens + [bos]
        else:
            return [bos] + raw_tokens + [eos]

    def __call__(self, batch: Union[str, list, tuple]):
        """Recursively search for strings in a list and wrap them in special tokens

        Args:
            batch: a string or a list containing strings

        Returns:
            the same structure where all strings are wrapped in special tokens
        """

        if isinstance(batch, (list, tuple)):
            batch = batch.copy()
        
        if self.tokenized:
            if isinstance(batch, (list, tuple)):
                if isinstance(batch[-1], str):
                    return self._wrap(batch, self.reverse, self.bos, self.eos)
                else:
                    return [self(line) for line in batch]
            raise RuntimeError(f'The objects passed to the reverser are not list or tuple! '
                               f' But they are {type(batch)}.'
                               f' If you want to passed str type directly use option tokenized = False')
        else:
            if isinstance(batch, (list, tuple)):
                return [self(line) for line in batch]
            else:
                return ' '.join(self._wrap(batch.split(), self.reverse, self.bos, self.eos))


@register('elmo_preprocessor')
class ELMoPreprocessor(Component):
    """Preparation of batches

    Args:
        batch_size: Size of a output batch.
        unroll_steps: Number of time steps of a output batch.
        max_word_length: Max length of words of input and output batches.
        n_gpus: Number of gpu to use.
    """
    def __init__(self, 
                 batch_size:int = 128,  
                 unroll_steps:int = 20,  
                 max_word_length:int = 50,  
                 n_gpus:int = 3,  
                 *args, 
                 **kwargs):
        self.batch_size = batch_size
        self.num_steps = unroll_steps
        self.max_word_length = max_word_length
        self.n_gpus = n_gpus

        # ================== Initialize a batching ==================
        self._batches_stream = {
            'reverse_stream':  {
                'stream_buffer': [None] * self.batch_size * self.n_gpus,
                'incompleted_batch': {
                    'init_batch_line_no': 0,
                    'line_pos': 0,
                    'char_ids': np.zeros([self.batch_size * self.n_gpus, self.num_steps, self.max_word_length],
                                        np.int32),
                    'token_ids': np.zeros([self.batch_size * self.n_gpus, self.num_steps], np.int32)
                },
                'raw_ids': {
                    'char_ids': [],
                    'token_ids': [],
                    'utilized': [],
                }
            },
            'stream': {
                'stream_buffer': [None] * self.batch_size * self.n_gpus,
                'incompleted_batch': {
                    'init_batch_line_no': 0,
                    'line_pos': 0,
                    'char_ids': np.zeros([self.batch_size * self.n_gpus, self.num_steps, self.max_word_length],
                                        np.int32),
                    'token_ids': np.zeros([self.batch_size * self.n_gpus, self.num_steps], np.int32)
                },
                'raw_ids': {
                    'char_ids': [],
                    'token_ids': [],
                    'utilized': [],
                }
            },
        }
    
    def __call__(self, char_ids: List[list], reversed_char_ids: List[list], token_ids: List[list], reversed_token_ids: List[list]):
        """
        This method is called by trainer to make one training step on one batch.

        Args:
            char_ids: batche of char_ids
            reversed_char_ids: batche of reversed_char_ids
            token_ids: batche of token_ids
            reversed_token_ids: batche of reversed_token_ids

        Returns:
             packed values of char_ids_batches, reversed_char_ids_batches, token_ids_batches, reversed_token_ids_batches
        """

        self._batches_stream['stream']['raw_ids']['char_ids'].extend(char_ids)
        self._batches_stream['stream']['raw_ids']['token_ids'].extend(token_ids)

        self._batches_stream['reverse_stream']['raw_ids']['char_ids'].extend(reversed_char_ids)
        self._batches_stream['reverse_stream']['raw_ids']['token_ids'].extend(reversed_token_ids)

        char_ids_batches, token_ids_batches = self._pack_batches(reverse = False)
        reversed_char_ids_batches, reversed_token_ids_batches = self._pack_batches(reverse = True)

        return char_ids_batches, reversed_char_ids_batches, token_ids_batches, reversed_token_ids_batches


    def _pack_batches(self, reverse: bool = False):
        batches_stream = self._batches_stream['reverse_stream'] if reverse else self._batches_stream['stream']
        generator = self._gen_batch_line(reverse, history = False)



        stream_buffer = batches_stream['stream_buffer']
        batch_line_no = None
        line_pos = None

        stop_and_wait_until_the_next_external_batch = False
        char_ids_batches = []
        token_ids_batches = []

        while True:
            if batch_line_no is None:
                batch_line_no = batches_stream['incompleted_batch']['init_batch_line_no']
                char_ids = batches_stream['incompleted_batch']['char_ids']
                token_ids = batches_stream['incompleted_batch']['token_ids']
            else:
                batch_line_no = 0
                char_ids = np.zeros([self.batch_size * self.n_gpus, self.num_steps, self.max_word_length], np.int32)
                token_ids = np.zeros([self.batch_size * self.n_gpus, self.num_steps], np.int32)

            for line_no in range(batch_line_no, self.batch_size * self.n_gpus):
                line_pos = batches_stream['incompleted_batch']['line_pos'] if line_pos is None else 0
                while line_pos < self.num_steps:
                    if stream_buffer[line_no] is None or len(stream_buffer[line_no][0]) <= 1:
                        try:
                            stream_buffer[line_no] = list(next(generator))
                        except StopIteration:
                            batches_stream['incompleted_batch']['init_batch_line_no'] = line_no
                            batches_stream['incompleted_batch']['line_pos'] = line_pos
                            batches_stream['incompleted_batch']['char_ids'] = char_ids
                            batches_stream['incompleted_batch']['token_ids'] = token_ids
                            batches_stream['stream_buffer'] = stream_buffer

                            stop_and_wait_until_the_next_external_batch = True

                            break

                    how_many = min(len(stream_buffer[line_no] [0]) - 1, self.num_steps - line_pos)
                    next_pos = line_pos + how_many
                    char_ids[line_no, line_pos:next_pos] = stream_buffer[line_no][0][:how_many]
                    token_ids[line_no, line_pos:next_pos] = stream_buffer[line_no][1][1:how_many+1]

                    line_pos = next_pos

                    stream_buffer[line_no][0] = stream_buffer[line_no][0][how_many:]
                    stream_buffer[line_no][1] = stream_buffer[line_no][1][how_many:]

                if stop_and_wait_until_the_next_external_batch: 
                    break
        
            if stop_and_wait_until_the_next_external_batch: 
                break
            else:
                char_ids_batches.append(char_ids)
                token_ids_batches.append(token_ids)
        return char_ids_batches, token_ids_batches

    def _gen_batch_line(self, reverse:bool = False, batch_no:int = 0, line_no:int = 0, history: bool = False):
        if reverse:
            raw_ids = self._batches_stream['reverse_stream']['raw_ids']
        else:
            raw_ids = self._batches_stream['stream']['raw_ids']
        if raw_ids['char_ids']:
            for char_ids, token_ids in zip(raw_ids['char_ids'],raw_ids['token_ids']):
                if history:
                    batch_count = round(len(char_ids) / self._options['unroll_steps'])
                    raw_ids['utilized'].append((batch_no, line_no, batch_count))
                yield char_ids, token_ids
            raw_ids['char_ids'] = [] 
            raw_ids['token_ids'] = []

@register('elmo_simple_preprocessor')
class ELMoSimplePreprocessor(Component):
    """Component for joining a list of strings.

    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch: List[List[str]]):
        """Join a list of strings.

        Args:
            batch: a list of lists containing strings

        Returns:
            a list of strings

        """
        return [" ".join(line) for line in batch]
