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

import sys
from overrides import overrides
from typing import List
import tensorflow as tf

import numpy as np
from bilm import Batcher as File_Batcher
from bilm import BidirectionalLanguageModel
from bilm import weight_layers
from deeppavlov.models.embedders.elmo.ext_batcher import ExtBatcher
from deeppavlov.models.embedders.elmo.vocabulary_creator import create_vocab

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.data.utils import zero_pad

log = get_logger(__name__)


@register('elmo')
class ELMoEmbedder(Component, Serializable):
    """
    Class implements ELMo embedding model
    """
    def __init__(self, load_path, save_path=None, dim=1024, pad_zero=False,
                                vocab_workers_n=2, chunk_size = 8, **kwargs):
        """
        Initialize embedder with given parameters
        Args:
            load_path: path where to load pre-trained embedding model and options file from
            save_path: is not used because model is not trainable; therefore, it is unchangable
            dim: dimensionality of ELMo model
            pad_zero: whether to pad samples or not
            **kwargs: additional arguments
        """
        super().__init__(save_path=save_path, load_path=load_path)
        self.tok2emb = {}
        self.dim = dim
        self.pad_zero = pad_zero
        self.char_per_token = 50
        self.vocab_workers_n = vocab_workers_n
        self.chunk_size = chunk_size
        self.character_ids, self.elmo_output, self.loaded_batcher = self.load()

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """
        Load ELMo binary model from self.load_path
        Args:
            *args: arguments
            **kwargs: arguments

        Returns:
            ELMo pre-trained model
        """
        if self.load_path and self.load_path.is_dir():
            log.info("[loading embeddings from `{}`]".format(self.load_path))
        else:
            log.error('No pretrained fasttext model provided or provided load_path "{}" is incorrect.'
                      .format(self.load_path))
            sys.exit(1)

        load_path2weights = self.load_path / 'weights.hdf5'
        load_path2options = self.load_path / 'options.json'
        load_path2vocab = self.load_path / 'vocab.txt'
        if load_path2weights.is_file():
            log.info("[loading elmo weights from `{}`]".format(load_path2weights))
        else:
            log.error('No pretrained elmo weights provided or provided load_path "{}" is incorrect.'
                      .format(self.load_path))
            sys.exit(1)

        if load_path2options.is_file():
            log.info("[loading elmo options from `{}`]".format(load_path2options))
        else:
            log.error('No elmo options provided or provided load_path "{}" is incorrect.'
                      .format(self.load_path))
            sys.exit(1)
        if load_path2vocab.is_file():
            log.info("[loading a vocab from `{}`]".format(load_path2vocab))
            loaded_batcher = File_Batcher(load_path2vocab, self.char_per_token)
        else:
            loaded_batcher = None


        # Input placeholders to the biLM.
        character_ids = tf.placeholder('int32', shape=(None, None, self.char_per_token))
        # Build the biLM graph.
        bilm = BidirectionalLanguageModel(str(load_path2options),
                    str(load_path2weights))

        # Get ops to compute the LM embeddings.
        embeddings_op = bilm(character_ids)

        # Get an op to compute ELMo (weighted average of the internal biLM layers)
        elmo_output = weight_layers('elmo_output', embeddings_op, l2_coef=0.0)

        return character_ids, elmo_output, loaded_batcher

    @overrides
    def __call__(self, batch, mean=False, *args, **kwargs):
        """
        Embed sentences from batch
        Args:
            batch: list of tokenized text samples
            mean: whether to return mean embedding of tokens per sample
            *args: arguments
            **kwargs: arguments

        Returns:
            embedded batch
        """
        if self.loaded_batcher:
            batcher = self.loaded_batcher
        else:
            vocab = create_vocab(batch, worker_n=self.vocab_workers_n, min_line_per_worker=10000)
            batcher = ExtBatcher(vocab, self.char_per_token)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        chunk_gen = self._chunk_generator(batch, self.chunk_size)
        # Create batches of data.
        data_ids = []
        sent_lens = []
        for chunked_data in chunk_gen:
            data_ids.append(batcher.batch_sentences(chunked_data))
            for y in chunked_data:
                sent_lens.append(len(y))

        # Compute ELMo representations (here for the input only, for simplicity).
        elmo_outputs = []
        for batch_ids in data_ids:
            elmo_outputs.extend(sess.run(self.elmo_output['weighted_op'],
                                              feed_dict={self.character_ids: batch_ids}))
        sess.close()

        for i, x in enumerate(elmo_outputs):
            elmo_outputs[i] = x[:sent_lens[i]]

        batch = elmo_outputs
        if mean:
            averaged_elmo_outputs = []
            for embedded_tokens in elmo_outputs:
                filtered = [et for et in embedded_tokens if np.any(et)]
                if filtered:
                    averaged_elmo_outputs.append(np.mean(filtered, axis=0))
                else:
                    averaged_elmo_outputs.append(np.zeros(self.dim, dtype=np.float32))
            batch = averaged_elmo_outputs

        if self.pad_zero:
            batch = zero_pad(batch)
        return batch

    def _chunk_generator(self, item_list, chunk_size):
        for i in range(0, len(item_list), chunk_size):
            yield item_list[i:i + chunk_size]

    def __iter__(self):
        """
        Iterate over all words from ELMo model vocabulary
        Returns:
            iterator
        """

        if self.loaded_batcher:
            yield from self.loaded_batcher._lm_vocab._id_to_word
        else:
            yield from ['<S>', '</S>', '<UNK>']
