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
from typing import List, Optional, Union, Tuple

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Dense, Lambda, Concatenate, Conv2D, Dropout, LSTM, Bidirectional,
                                     TimeDistributed)
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.regularizers import l2

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.core.models.keras_model import KerasModel
from .cells import Highway
from .common_tagger import to_one_hot

log = getLogger(__name__)

MAX_WORD_LENGTH = 30


@register("morpho_tagger")
class MorphoTagger(KerasModel):
    """A class for character-based neural morphological tagger

    Parameters:
        symbols: character vocabulary
        tags: morphological tags vocabulary
        save_path: the path where model is saved
        load_path: the path from where model is loaded
        mode: usage mode

        word_rnn: the type of character-level network (only `cnn` implemented)
        char_embeddings_size: the size of character embeddings
        char_conv_layers: the number of convolutional layers on character level
        char_window_size: the width of convolutional filter (filters).
            It can be a list if several parallel filters are applied, for example, [2, 3, 4, 5].
        char_filters: the number of convolutional filters for each window width.
            It can be a number, a list (when there are several windows of different width
            on a single convolution layer), a list of lists, if there
            are more than 1 convolution layers, or **None**.
            If **None**, a layer with width **width** contains
            min(**char_filter_multiple** * **width**, 200) filters.

        char_filter_multiple: the ratio between filters number and window width
        char_highway_layers: the number of highway layers on character level
        conv_dropout: the ratio of dropout between convolutional layers
        highway_dropout: the ratio of dropout between highway layers,
        intermediate_dropout: the ratio of dropout between convolutional
            and highway layers on character level
        lstm_dropout: dropout ratio in word-level LSTM
        word_vectorizers: list of parameters for additional word-level vectorizers,
            for each vectorizer it stores a pair of vectorizer dimension and
            the dimension of the corresponding word embedding
        word_lstm_layers: the number of word-level LSTM layers
        word_lstm_units: hidden dimensions of word-level LSTMs
        word_dropout: the ratio of dropout before word level (it is applied to word embeddings)
        regularizer: l2 regularization parameter
        verbose: the level of verbosity

    A subclass of :class:`~deeppavlov.core.models.keras_model.KerasModel`
    """
    def __init__(self,
                 symbols: SimpleVocabulary,
                 tags: SimpleVocabulary,
                 save_path: Optional[Union[str, Path]] = None,
                 load_path: Optional[Union[str, Path]] = None,
                 mode: str = 'infer',
                 word_rnn: str = "cnn",
                 char_embeddings_size: int = 16,
                 char_conv_layers: int = 1,
                 char_window_size: Union[int, List[int]] = 5,
                 char_filters: Union[int, List[int]] = None,
                 char_filter_multiple: int = 25,
                 char_highway_layers: int = 1,
                 conv_dropout: float = 0.0,
                 highway_dropout: float = 0.0,
                 intermediate_dropout: float = 0.0,
                 lstm_dropout: float = 0.0,
                 word_vectorizers: List[Tuple[int, int]] = None,
                 word_lstm_layers: int = 1,
                 word_lstm_units: Union[int, List[int]] = 128,
                 word_dropout: float = 0.0,
                 regularizer: float = None,
                 verbose: int = 1, **kwargs):
        # Calls parent constructor. Results in creation of save_folder if it doesn't exist
        super().__init__(save_path=save_path, load_path=load_path, mode=mode, **kwargs)
        self.symbols = symbols
        self.tags = tags
        self.word_rnn = word_rnn
        self.char_embeddings_size = char_embeddings_size
        self.char_conv_layers = char_conv_layers
        self.char_window_size = char_window_size
        self.char_filters = char_filters
        self.char_filter_multiple = char_filter_multiple
        self.char_highway_layers = char_highway_layers
        self.conv_dropout = conv_dropout
        self.highway_dropout = highway_dropout
        self.intermediate_dropout = intermediate_dropout
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.word_vectorizers = word_vectorizers  # a list of additional vectorizer dimensions
        self.word_lstm_layers = word_lstm_layers
        self.word_lstm_units = word_lstm_units
        self.regularizer = regularizer
        self.verbose = verbose
        self._initialize()
        self.model_ = None
        self.build()

        # Tries to load the model from model `load_path`, if it is available
        self.load()

    def load(self) -> None:
        """
        Checks existence of the model file, loads the model if the file exists
        Loads model weights from a file
        """

        # Checks presence of the model files
        if self.load_path.exists():
            path = str(self.load_path.resolve())
            log.info('[loading model from {}]'.format(path))
            self.model_.load_weights(path)

    def save(self) -> None:
        """
        Saves model weights to the save_path, provided in config. The directory is
        already created by super().__init__, which is called in __init__ of this class"""
        path = str(self.save_path.absolute())
        log.info('[saving model to {}]'.format(path))
        self.model_.save_weights(path)

    def _initialize(self):
        if isinstance(self.char_window_size, int):
            self.char_window_size = [self.char_window_size]
        if self.char_filters is None or isinstance(self.char_filters, int):
            self.char_filters = [self.char_filters] * len(self.char_window_size)
        if len(self.char_window_size) != len(self.char_filters):
            raise ValueError("There should be the same number of window sizes and filter sizes")
        if isinstance(self.word_lstm_units, int):
            self.word_lstm_units = [self.word_lstm_units] * self.word_lstm_layers
        if len(self.word_lstm_units) != self.word_lstm_layers:
            raise ValueError("There should be the same number of lstm layer units and lstm layers")
        if self.word_vectorizers is None:
            self.word_vectorizers = []
        if self.regularizer is not None:
            self.regularizer = l2(self.regularizer)
        if self.verbose > 0:
            log.info("{} symbols, {} tags in CharacterTagger".format(len(self.symbols), len(self.tags)))

    def build(self):
        """Builds the network using Keras.
        """
        word_inputs = Input(shape=(None, MAX_WORD_LENGTH+2), dtype="int32")
        inputs = [word_inputs]
        word_outputs = self._build_word_cnn(word_inputs)
        if len(self.word_vectorizers) > 0:
            additional_word_inputs = [Input(shape=(None, input_dim), dtype="float32")
                                      for input_dim, dense_dim in self.word_vectorizers]
            inputs.extend(additional_word_inputs)
            additional_word_embeddings = [Dense(dense_dim)(additional_word_inputs[i])
                                          for i, (_, dense_dim) in enumerate(self.word_vectorizers)]
            word_outputs = Concatenate()([word_outputs] + additional_word_embeddings)
        outputs, lstm_outputs = self._build_basic_network(word_outputs)
        compile_args = {"optimizer": Nadam(lr=0.002, clipnorm=5.0),
                        "loss": "categorical_crossentropy", "metrics": ["accuracy"]}
        self.model_ = Model(inputs, outputs)
        self.model_.compile(**compile_args)
        if self.verbose > 0:
            self.model_.summary(print_fn=log.info)
        return self

    def _build_word_cnn(self, inputs):
        """Builds word-level network
        """
        inputs = Lambda(K.one_hot, arguments={"num_classes": len(self.symbols)},
                        output_shape=lambda x: tuple(x) + (len(self.symbols),))(inputs)
        char_embeddings = Dense(self.char_embeddings_size, use_bias=False)(inputs)
        conv_outputs = []
        self.char_output_dim_ = 0
        for window_size, filters_number in zip(self.char_window_size, self.char_filters):
            curr_output = char_embeddings
            curr_filters_number = (min(self.char_filter_multiple * window_size, 200)
                                   if filters_number is None else filters_number)
            for _ in range(self.char_conv_layers - 1):
                curr_output = Conv2D(curr_filters_number, (1, window_size),
                                     padding="same", activation="relu",
                                     data_format="channels_last")(curr_output)
                if self.conv_dropout > 0.0:
                    curr_output = Dropout(self.conv_dropout)(curr_output)
            curr_output = Conv2D(curr_filters_number, (1, window_size),
                                 padding="same", activation="relu",
                                 data_format="channels_last")(curr_output)
            conv_outputs.append(curr_output)
            self.char_output_dim_ += curr_filters_number
        if len(conv_outputs) > 1:
            conv_output = Concatenate(axis=-1)(conv_outputs)
        else:
            conv_output = conv_outputs[0]
        highway_input = Lambda(K.max, arguments={"axis": -2})(conv_output)
        if self.intermediate_dropout > 0.0:
            highway_input = Dropout(self.intermediate_dropout)(highway_input)
        for i in range(self.char_highway_layers - 1):
            highway_input = Highway(activation="relu")(highway_input)
            if self.highway_dropout > 0.0:
                highway_input = Dropout(self.highway_dropout)(highway_input)
        highway_output = Highway(activation="relu")(highway_input)
        return highway_output

    def _build_basic_network(self, word_outputs):
        """
        Creates the basic network architecture,
        transforming word embeddings to intermediate outputs
        """
        if self.word_dropout > 0.0:
            lstm_outputs = Dropout(self.word_dropout)(word_outputs)
        else:
            lstm_outputs = word_outputs
        for j in range(self.word_lstm_layers-1):
            lstm_outputs = Bidirectional(
                LSTM(self.word_lstm_units[j], return_sequences=True,
                     dropout=self.lstm_dropout))(lstm_outputs)
        lstm_outputs = Bidirectional(
                LSTM(self.word_lstm_units[-1], return_sequences=True,
                     dropout=self.lstm_dropout))(lstm_outputs)
        pre_outputs = TimeDistributed(
                Dense(len(self.tags), activation="softmax",
                      activity_regularizer=self.regularizer),
                name="p")(lstm_outputs)
        return pre_outputs, lstm_outputs

    # noinspection PyPep8Naming
    def _transform_batch(self, data, labels=None, transform_to_one_hot=True):
        data, additional_data = data[0], data[1:]
        L = max(len(x) for x in data)
        X = np.array([self._make_sent_vector(x, L) for x in data])
        X = [X] + [np.array(x) for x in additional_data]
        if labels is not None:
            Y = np.array([self._make_tags_vector(y, L) for y in labels])
            if transform_to_one_hot:
                Y = to_one_hot(Y, len(self.tags))
            return X, Y
        else:
            return X

    def train_on_batch(self, *args) -> None:
        """Trains the model on a single batch.

        Args:
            *args: the list of network inputs. Last element of `args` is the batch of targets,
                all previous elements are training data batches
        """
        # data: List[Iterable], labels: Iterable[list]
        # Args:
        #   data: a batch of word sequences
        #   labels: a batch of correct tag sequences
        *data, labels = args
        # noinspection PyPep8Naming
        X, Y = self._transform_batch(data, labels)
        self.model_.train_on_batch(X, Y)

    # noinspection PyPep8Naming
    def predict_on_batch(self, data: Union[List[np.ndarray], Tuple[np.ndarray]],
                         return_indexes: bool = False) -> List[List[str]]:
        """
        Makes predictions on a single batch

        Args:
            data: model inputs for a single batch, data[0] contains input character encodings
            and is the only element of data for mist models. Subsequent elements of data
            include the output of additional vectorizers, e.g., dictionary-based one.
            return_indexes: whether to return tag indexes in vocabulary or the tags themselves

        Returns:
            a batch of label sequences
        """
        X = self._transform_batch(data)
        objects_number, lengths = len(X[0]), [len(elem) for elem in data[0]]
        Y = self.model_.predict_on_batch(X)
        labels = np.argmax(Y, axis=-1)
        answer: List[Optional[List[str]]] = [None] * objects_number
        for i, (elem, length) in enumerate(zip(labels, lengths)):
            elem = elem[:length]
            answer[i] = elem if return_indexes else self.tags.idxs2toks(elem)
        return answer

    def __call__(self, *x_batch: np.ndarray, **kwargs) -> Union[List, np.ndarray]:
        """
        Predicts answers on batch elements.

        Args:
            x_batch: a batch to predict answers on. It can be either a single array
                for basic model or a sequence of arrays for a complex one (
                :config:`configuration file <morpho_tagger/UD2.0/morpho_ru_syntagrus_pymorphy.json>`
                or its lemmatized version).
        """
        return self.predict_on_batch(x_batch, **kwargs)

    def _make_sent_vector(self, sent: List, bucket_length: int = None) -> np.ndarray:
        """Transforms a sentence to Numpy array, which will be the network input.

        Args:
            sent: input sentence
            bucket_length: the width of the bucket

        Returns:
            A 3d array, answer[i][j][k] contains the index of k-th letter
            in j-th word of i-th input sentence.
        """
        bucket_length = bucket_length or len(sent)
        answer = np.zeros(shape=(bucket_length, MAX_WORD_LENGTH+2), dtype=np.int32)
        for i, word in enumerate(sent):
            answer[i, 0] = self.tags["BEGIN"]
            m = min(len(word), MAX_WORD_LENGTH)
            for j, x in enumerate(word[-m:]):
                answer[i, j+1] = self.symbols[x]
            answer[i, m+1] = self.tags["END"]
            answer[i, m+2:] = self.tags["PAD"]
        return answer

    def _make_tags_vector(self, tags, bucket_length=None) -> np.ndarray:
        """Transforms a sentence of tags to Numpy array, which will be the network target.

        Args:
            tags: input sentence of tags
            bucket_length: the width of the bucket

        Returns:
            A 2d array, answer[i][j] contains the index of j-th tag in i-th input sentence.
        """
        bucket_length = bucket_length or len(tags)
        answer = np.zeros(shape=(bucket_length,), dtype=np.int32)
        for i, tag in enumerate(tags):
            answer[i] = self.tags[tag]
        return answer
