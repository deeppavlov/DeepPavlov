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

import sys
from typing import Iterator, List, Union, Optional


import json
import numpy as np
import tensorflow as tf
from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.tf_backend import TfModelMeta

from deeppavlov.models.bidirectional_lms.elmo.utils import load_model, load_options_latest_checkpoint
from deeppavlov.models.bidirectional_lms.elmo.data import InferBatcher
from logging import getLogger

logger = getLogger(__name__)

@register('elmo_bilm')
class ELMoEmbedder(Component, metaclass=TfModelMeta):
    """

    """
    def __init__(self, model_dir: str, forward_direction_sequence: bool = True, backward_direction_sequence: bool = True,
                 pad_zero: bool = False, max_token: Optional[int] = None, mini_batch_size: int = 32, **kwargs) -> None:

        self.model_dir = model_dir if '://' in model_dir else str(expand_path(model_dir))

        self.forward_direction_sequence = forward_direction_sequence
        self.backward_direction_sequence = backward_direction_sequence
        if not (self.forward_direction_sequence or self.backward_direction_sequence):
            log.error(f'At least one direction sequence of forward_direction_sequence or backward_direction_sequence'\
                      ' must be equal to True.')
            sys.exit(1)

        self.pad_zero = pad_zero
        self.max_token = max_token
        self.mini_batch_size = mini_batch_size
        self.model, self.sess, self.init_states, self.batcher, self.options = self._load()

    def _load(self):
        """

        Returns:
        """

        options, ckpt_file, vocab_file = load_options_latest_checkpoint(self.model_dir)

        max_token_length = options['char_cnn']['max_characters_per_token']
        batcher = InferBatcher(vocab_file, max_token_length)

        model, sess, init_state_tensors, init_state_values, final_state_tensors = load_model(options, ckpt_file, self.mini_batch_size)

        init_states = (init_state_tensors, init_state_values, final_state_tensors)

        return model, sess, init_states, batcher, options

    def _fill_batch(self, batch):
        """
        Fill batch correct values.

        Args:
            batch: A list of tokenized text samples.

        Returns:
            batch: A list of tokenized text samples.
        """

        if not batch:
            log.Warning('Need implementation')

        filled_batch = []
        for batch_line in batch:
            batch_line = batch_line if batch_line else ['']
            filled_batch.append(batch_line)

        batch = filled_batch

        if self.max_token:
            batch = [batch_line[:self.max_token] for batch_line in batch]
        tokens_length = [len(batch_line) for batch_line in batch]
        tokens_length_max = max(tokens_length)
        batch_notreverse = [batch_line + ['']*(tokens_length_max - len(batch_line)) for batch_line in batch]
        batch_reverse = [list(reversed(batch_line)) + ['']*(tokens_length_max - len(batch_line)) for batch_line in batch]

        return batch_notreverse, batch_reverse, tokens_length

    def _mini_batch_fit(self, batch: List[List[str]], init_state_tensors, init_state_values, final_state_tensors,
                 *args, **kwargs) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed sentences from a batch.

        Args:
            batch: A list of tokenized text samples.
            init_state_tensors: ----.
            init_state_values: ----.
            final_state_tensors: ----.

        Returns:
            A mini batch of lm predictions.
        """
        batch, batch_reverse, tokens_length = self._fill_batch(batch)


        # time major
        batch = np.expand_dims(self.batcher.batch_sentences(batch).transpose(1,0,2), axis=2)
        batch_reverse = np.expand_dims(self.batcher.batch_sentences(batch_reverse).transpose(1,0,2), axis=2)

        pad_size = self.mini_batch_size - batch.shape[1]

        #time iterations
        output_batch = np.zeros((batch.shape[0],batch.shape[1],self.options['n_tokens_vocab']))
        output_batch_reverse = np.zeros((batch.shape[0],batch.shape[1],self.options['n_tokens_vocab']))
        for batch_no, (time_sliced_batch, time_sliced_batch_reverse) in enumerate(zip (batch,batch_reverse)):

            #batch padding
            complete_batch = np.pad(time_sliced_batch, ((0,pad_size),(0,0),(0,0)), 'constant',constant_values=260)
            complete_batch_reverse = np.pad(time_sliced_batch_reverse, ((0,pad_size),(0,0),(0,0)), 'constant',constant_values=260)

            feed_dict = {t: v for t, v in zip(init_state_tensors, init_state_values)}
            feed_dict[self.model.tokens_characters] = complete_batch
            feed_dict[self.model.tokens_characters_reverse] = complete_batch_reverse

            ret = self.sess.run([self.model.individual_output_softmaxes, final_state_tensors],
                                feed_dict=feed_dict
                                )
            individual_output_softmaxes, init_state_values = ret
            
            #remove padded parts of a batch and save in a share matrix
            output_batch[batch_no] = individual_output_softmaxes[0][:batch.shape[1]]
            output_batch_reverse[batch_no] =  individual_output_softmaxes[1][:batch.shape[1]]

        # remove a prediction of </S> and next token
        output_batch = output_batch[:-2]
        output_batch_reverse = output_batch_reverse[:-2]

        # batch major
        output_batch = output_batch.transpose(1,0,2)
        output_batch_reverse = output_batch_reverse.transpose(1,0,2)

        # remove pads of time and reverse a reverse
        output_batch = [batch_line[:tok_len] for batch_line, tok_len in zip(output_batch, tokens_length)]
        output_batch_reverse = [np.flip(batch_line[:tok_len],axis=0) for batch_line, tok_len in zip(output_batch_reverse, tokens_length)]
        
        output_full_batch = []
        for batch_line, batch_line_reverse, tok_len in zip(output_batch, output_batch_reverse, tokens_length):
            line = np.concatenate((batch_line,batch_line_reverse), axis=-1)
            # [time x 2*vocab_size] -> [time x 2 x vocab_size]
            line = np.reshape(line, (tok_len, -1,  self.options['n_tokens_vocab']))
            output_full_batch.append(line)


        return output_full_batch, init_state_values


    @staticmethod
    def chunk_generator(items_list, chunk_size):
        for i in range(0, len(items_list), chunk_size):
                        yield items_list[i:i + chunk_size]
    @overrides
    def __call__(self, batch: List[List[str]],
                 *args, **kwargs) -> Union[List[np.ndarray], np.ndarray]:
        """

        Args:
            batch: A list of tokenized text samples.

        Returns:
            A  batch of lm predictions.
        """

        init_state_tensors, init_state_values, final_state_tensors = self.init_states

        if len(batch) > self.mini_batch_size:
            batch_gen = self.chunk_generator(batch, self.mini_batch_size)
            output_batch = []
            for mini_batch in batch_gen:
                mini_batch_out, init_state_values = self._mini_batch_fit(mini_batch, 
                                                                            init_state_tensors, init_state_values, 
                                                                            final_state_tensors, *args, **kwargs)
                output_batch.extend(mini_batch_out)
        else:
            output_batch, init_state_values = self._mini_batch_fit(batch, 
                                                                            init_state_tensors, init_state_values, 
                                                                            final_state_tensors, *args, **kwargs)
        
        self.init_states = (init_state_tensors, init_state_values, final_state_tensors)
        return output_batch

    def get_vocab_size(self) -> int:
        """
        
        Returns:
            vocab size
        """

        return self.options['n_tokens_vocab']
    
    def get_vocab(self) -> list:
        """
        
        Returns:
            vocab size
        """

        return self.batcher._lm_vocab._id_to_word


    def destroy(self):
        self.sess.close()
