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
import tensorflow_hub as hub

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.data.utils import zero_pad

log = get_logger(__name__)


@register('elmo')
class ELMoEmbedder(Component, Serializable):
    """
    ``ELMo`` (Embeddings from Language Models) representations are pre-trained contextual representations from large scale bidirectional language models.
    Parameters:
        spec: A ModuleSpec defining the Module to instantiate or a path where to load a ModuleSpec from via load_module_spec by using tenserflow_hub.
        dim: Dimensionality of output token embeddings of ELMo model.
        char_emb_dim: Dimensionality of token embeddings.
        pad_zero: Whether to use pad samples or not.
        load_path: Load path is not used.
        save_path: Save path is not used.
    """
    def __init__(self, spec, dim=1024, pad_zero=False, load_path=None, save_path=None, **kwargs):

        super().__init__(save_path=save_path, load_path=load_path)
        self.spec = spec
        self.dim = dim
        self.pad_zero = pad_zero
        self.elmo_module = self.load()

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """
        Load a ELMo tensorflow hub module from a self.spec.
        Args:
            *args: arguments.
            **kwargs: arguments.

        Returns:
            A ELMo pre-trained model is wrapped a tenserflow hub module.
        """
        elmo_module = hub.Module(self.spec, trainable=False)

        return elmo_module

    @overrides
    def __call__(self, batch, mean=False, *args, **kwargs):
        """
        Embed sentences from a batch.
        Args:
            batch: A list of tokenized text samples.
            mean: Whether to return a mean ELMo embedding of tokens per sample.
        Returns:
            a batch of ELMo embeddings.
        """
        if not batch:
            return batch

        tokens_length = [len(batch_line) for batch_line in batch]
        tokens_length_max = max(tokens_length)
        batch = [batch_line + ['']*(tokens_length_max - len(batch_line)) for batch_line in batch]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        elmo_outputs = self.elmo_module(
                                        inputs={
                                            "tokens": batch,
                                            "sequence_len": tokens_length
                                        },
                                        signature="tokens",
                                        as_dict=True)

        elmo_outputs = sess.run(elmo_outputs)
        sess.close()

        if mean:
            batch = elmo_outputs['default']

            dim0, dim1 = batch.shape

            if self.dim != dim1:
                batch = np.resize(batch, (dim0,self.dim))
        else:
            batch = elmo_outputs['elmo']

            dim0, dim1, dim2 = batch.shape

            if self.dim != dim2:
                batch = np.resize(batch, (dim0, dim1,self.dim))

            batch = [batch_line[:length_line] for length_line, batch_line in zip(tokens_length,batch)]

            if self.pad_zero:
                batch = zero_pad(batch)

        return batch

    def __iter__(self):
        """
        Iterate over all words from a ELMo model vocabulary.
        The ELMo model vocabulary consists of '<S>', '</S>', '<UNK>'.
        Returns:
            An iterator of three elements ('<S>', '</S>', '<UNK>').
        """

        yield from ['<S>', '</S>', '<UNK>']
