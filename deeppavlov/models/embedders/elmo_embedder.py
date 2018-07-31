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

log = get_logger(__name__)


@register('elmo')
class ELMoEmbedder(Component, Serializable):
    """
    Class implements ELMo embedding model
    """
    def __init__(self, spec load_path=None, save_path=None, dim=1024, dim=1024, **kwargs):
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
        self.spec = spec
        self.dim = dim
        self.elmo_module = self.load()

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """
        Load ELMo tensorflow hub module from self.spec
        Args:
            *args: arguments
            **kwargs: arguments

        Returns:
            ELMo pre-trained model
        """
        elmo_module = hub.Module(self.spec, trainable=False)

        return elmo_module

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
        tokens_length = [len(batch_line) for batch_line in batch]
        embeddings = elmo(
            inputs={
                "tokens": batch,
                "sequence_len": tokens_length
            },
            signature="tokens",
            as_dict=True)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        elmo_outputs = self.elmo_module(
                                        inputs={
                                            "tokens": batch,
                                            "sequence_len": tokens_length
                                        },
                                        signature="tokens",
                                        as_dict=True)
        sess.close()

        if mean:
            batch = elmo_outputs['default']
        else:
            batch = elmo_outputs['elmo']

        dim1, dim2 = batch.shape
        if self.dim != dim2:
            batch = np.resize(batch, (dim1,self.dim))

        return batch

    def __iter__(self):
        """
        Iterate over all words from ELMo model vocabulary
        Returns:
            iterator
        """

        yield from ['<S>', '</S>', '<UNK>']
