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
from logging import getLogger
from typing import Iterator, List, Union, Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from overrides import overrides

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad, chunk_generator
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.tf_backend import TfModelMeta

log = getLogger(__name__)


@register('elmo_embedder')
class ELMoEmbedder(Component, metaclass=TfModelMeta):
    """
    ``ELMo`` (Embeddings from Language Models) representations are pre-trained contextual representations from
    large-scale bidirectional language models. See a paper `Deep contextualized word representations
    <https://arxiv.org/abs/1802.05365>`__ for more information about the algorithm and a detailed analysis.

    Parameters:
        spec: A ``ModuleSpec`` defining the Module to instantiate or a path where to load a ``ModuleSpec`` from via
            ``tenserflow_hub.load_module_spec`` by using `TensorFlow Hub <https://www.tensorflow.org/hub/overview>`__.
        elmo_output_names: A list of output ELMo. You can use combination of
            ``["word_emb", "lstm_outputs1", "lstm_outputs2","elmo"]`` and you can use separately ``["default"]``.

            Where,

            * ``word_emb`` - CNN embedding (default dim 512)
            * ``lstm_outputs*`` - ouputs of lstm (default dim 1024)
            * ``elmo`` - weighted sum of cnn and lstm outputs (default dim 1024)
            * ``default`` - mean ``elmo`` vector for sentence (default dim 1024)

            See `TensorFlow Hub <https://www.tensorflow.org/hub/modules/google/elmo/2>`__ for more information about it.
        dim: Can be used for output embeddings dimensionality reduction if elmo_output_names != ['default']
        pad_zero: Whether to use pad samples or not.
        concat_last_axis: A boolean that enables/disables last axis concatenation. It is not used for
            ``elmo_output_names = ["default"]``.
        max_token: The number limitation of words per a batch line.
        mini_batch_size: It is used to reduce the memory requirements of the device.


    If some required packages are missing, install all the requirements by running in command line:

    .. code:: bash

        python -m deeppavlov install <path_to_config>

    where ``<path_to_config>`` is a path to one of the :config:`provided config files <elmo_embedder>`
    or its name without an extension, for example :

    .. code:: bash

        python -m deeppavlov install elmo_ru-news
        
    Examples:
        >>> from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
        >>> elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz")
        >>> elmo([['вопрос', 'жизни', 'Вселенной', 'и', 'вообще', 'всего'], ['42']])
        array([[ 0.00719104,  0.08544601, -0.07179783, ...,  0.10879009,
                -0.18630421, -0.2189409 ],
               [ 0.16325025, -0.04736076,  0.12354863, ..., -0.1889013 ,
                 0.04972512,  0.83029324]], dtype=float32)

        You can use ELMo models from DeepPavlov as usual `TensorFlow Hub Module
        <https://www.tensorflow.org/hub/modules/google/elmo/2>`_.

        >>> import tensorflow as tf
        >>> import tensorflow_hub as hub
        >>> elmo = hub.Module("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",
        trainable=True)
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> embeddings = elmo(["это предложение", "word"], signature="default", as_dict=True)["elmo"]
        >>> sess.run(embeddings)
        array([[[ 0.05817392,  0.22493343, -0.19202903, ..., -0.14448944,
                 -0.12425567,  1.0148407 ],
                [ 0.53596294,  0.2868537 ,  0.28028542, ..., -0.08028372,
                  0.49089077,  0.75939953]],
               [[ 0.3433637 ,  1.0031182 , -0.1597258 , ...,  1.2442509 ,
                  0.61029315,  0.43388373],
                [ 0.05370751,  0.02260921,  0.01074906, ...,  0.08748816,
                 -0.0066415 , -0.01344293]]], dtype=float32)

        TensorFlow Hub module also supports tokenized sentences in the following format.

        >>> tokens_input = [["мама", "мыла", "раму"], ["рама", "", ""]]
        >>> tokens_length = [3, 1]
        >>> embeddings = elmo(
                inputs={
                        "tokens": tokens_input,
                        "sequence_len": tokens_length
                        },
                signature="tokens",
                as_dict=True)["elmo"]
        >>> sess.run(embeddings)
        array([[[ 0.6040001 , -0.16130011,  0.56478846, ..., -0.00376141,
                 -0.03820051,  0.26321286],
                [ 0.01834148,  0.17055789,  0.5311495 , ..., -0.5675535 ,
                  0.62669843, -0.05939034],
                [ 0.3242596 ,  0.17909613,  0.01657108, ...,  0.1866098 ,
                  0.7392496 ,  0.08285746]],
               [[ 1.1322289 ,  0.19077688, -0.17811403, ...,  0.42973226,
                  0.23391506, -0.01294377],
                [ 0.05370751,  0.02260921,  0.01074906, ...,  0.08748816,
                 -0.0066415 , -0.01344293],
                [ 0.05370751,  0.02260921,  0.01074906, ...,  0.08748816,
                 -0.0066415 , -0.01344293]]], dtype=float32)

        You can also get ``hub.text_embedding_column`` like described `here
        <https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub#feature_columns>`_.


    """

    def __init__(self, spec: str, elmo_output_names: Optional[List] = None,
                 dim: Optional[int] = None, pad_zero: bool = False,
                 concat_last_axis: bool = True, max_token: Optional[int] = None,
                 mini_batch_size: int = 32, **kwargs) -> None:

        self.spec = spec if '://' in spec else str(expand_path(spec))

        self.elmo_output_dims = {'word_emb': 512,
                                 'lstm_outputs1': 1024,
                                 'lstm_outputs2': 1024,
                                 'elmo': 1024,
                                 'default': 1024}
        elmo_output_names = elmo_output_names or ['default']
        self.elmo_output_names = elmo_output_names
        elmo_output_names_set = set(self.elmo_output_names)
        if elmo_output_names_set - set(self.elmo_output_dims.keys()):
            log.error(f'Incorrect elmo_output_names = {elmo_output_names} . You can use either  ["default"] or some of'
                      '["word_emb", "lstm_outputs1", "lstm_outputs2","elmo"]')
            sys.exit(1)

        if elmo_output_names_set - {'default'} and elmo_output_names_set - {"word_emb", "lstm_outputs1",
                                                                            "lstm_outputs2", "elmo"}:
            log.error('Incompatible conditions: you can use either  ["default"] or list of '
                      '["word_emb", "lstm_outputs1", "lstm_outputs2","elmo"] ')
            sys.exit(1)

        self.pad_zero = pad_zero
        self.concat_last_axis = concat_last_axis
        self.max_token = max_token
        self.mini_batch_size = mini_batch_size
        self.elmo_outputs, self.sess, self.tokens_ph, self.tokens_length_ph = self._load()
        self.dim = self._get_dims(self.elmo_output_names, dim, concat_last_axis)

    def _get_dims(self, elmo_output_names, in_dim, concat_last_axis):
        dims = [self.elmo_output_dims[elmo_output_name] for elmo_output_name in elmo_output_names]
        if concat_last_axis:
            dims = in_dim if in_dim else sum(dims)
        else:
            if in_dim:
                log.warning(f"[ dim = {in_dim} is not used, because the elmo_output_names has more than one element.]")
        return dims

    def _load(self):
        """
        Load a ELMo TensorFlow Hub Module from a self.spec.

        Returns:
            ELMo pre-trained model wrapped in TenserFlow Hub Module.
        """
        elmo_module = hub.Module(self.spec, trainable=False)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        tokens_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='tokens')
        tokens_length_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='tokens_length')

        elmo_outputs = elmo_module(inputs={"tokens": tokens_ph,
                                           "sequence_len": tokens_length_ph},
                                   signature="tokens",
                                   as_dict=True)

        sess.run(tf.global_variables_initializer())

        return elmo_outputs, sess, tokens_ph, tokens_length_ph

    def _fill_batch(self, batch):
        """
        Fill batch correct values.

        Args:
            batch: A list of tokenized text samples.

        Returns:
            batch: A list of tokenized text samples.
        """

        if not batch:
            empty_vec = np.zeros(self.dim, dtype=np.float32)
            return [empty_vec] if 'default' in self.elmo_output_names else [[empty_vec]]

        filled_batch = []
        for batch_line in batch:
            batch_line = batch_line if batch_line else ['']
            filled_batch.append(batch_line)

        batch = filled_batch

        if self.max_token:
            batch = [batch_line[:self.max_token] for batch_line in batch]
        tokens_length = [len(batch_line) for batch_line in batch]
        tokens_length_max = max(tokens_length)
        batch = [batch_line + [''] * (tokens_length_max - len(batch_line)) for batch_line in batch]

        return batch, tokens_length

    def _mini_batch_fit(self, batch: List[List[str]], *args, **kwargs) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed sentences from a batch.

        Args:
            batch: A list of tokenized text samples.

        Returns:
            A batch of ELMo embeddings.
        """
        batch, tokens_length = self._fill_batch(batch)

        elmo_outputs = self.sess.run(self.elmo_outputs,
                                     feed_dict={self.tokens_ph: batch,
                                                self.tokens_length_ph: tokens_length})

        if 'default' in self.elmo_output_names:
            elmo_output_values = elmo_outputs['default']
            dim0, dim1 = elmo_output_values.shape
            if self.dim != dim1:
                shape = (dim0, self.dim if isinstance(self.dim, int) else self.dim[0])
                elmo_output_values = np.resize(elmo_output_values, shape)
        else:
            elmo_output_values = [elmo_outputs[elmo_output_name] for elmo_output_name in self.elmo_output_names]
            elmo_output_values = np.concatenate(elmo_output_values, axis=-1)

            dim0, dim1, dim2 = elmo_output_values.shape
            if self.concat_last_axis and self.dim != dim2:
                shape = (dim0, dim1, self.dim)
                elmo_output_values = np.resize(elmo_output_values, shape)

            elmo_output_values = [elmo_output_values_line[:length_line]
                                  for length_line, elmo_output_values_line in zip(tokens_length, elmo_output_values)]

            if not self.concat_last_axis:
                slice_indexes = np.cumsum(self.dim).tolist()[:-1]
                elmo_output_values = [[np.array_split(vec, slice_indexes) for vec in tokens]
                                      for tokens in elmo_output_values]

        return elmo_output_values

    @overrides
    def __call__(self, batch: List[List[str]],
                 *args, **kwargs) -> Union[List[np.ndarray], np.ndarray]:
        """
        Embed sentences from a batch.

        Args:
            batch: A list of tokenized text samples.

        Returns:
            A batch of ELMo embeddings.
        """
        if len(batch) > self.mini_batch_size:
            batch_gen = chunk_generator(batch, self.mini_batch_size)
            elmo_output_values = []
            for mini_batch in batch_gen:
                mini_batch_out = self._mini_batch_fit(mini_batch, *args, **kwargs)
                elmo_output_values.extend(mini_batch_out)
        else:
            elmo_output_values = self._mini_batch_fit(batch, *args, **kwargs)

        if self.pad_zero:
            elmo_output_values = zero_pad(elmo_output_values)

        return elmo_output_values

    def __iter__(self) -> Iterator:
        """
        Iterate over all words from a ELMo model vocabulary.
        The ELMo model vocabulary consists of ``['<S>', '</S>', '<UNK>']``.

        Returns:
            An iterator of three elements ``['<S>', '</S>', '<UNK>']``.
        """

        yield from ['<S>', '</S>', '<UNK>']

    def destroy(self):
        if hasattr(self, 'sess'):
            for k in list(self.sess.graph.get_all_collection_keys()):
                self.sess.graph.clear_collection(k)
        super().destroy()
