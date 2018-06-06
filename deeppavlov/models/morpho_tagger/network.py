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

import inspect
import json
from typing import List

import keras.layers as kl
import keras.optimizers as ko
import keras.regularizers as kreg
from keras import Model

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.data.vocab import DefaultVocabulary
from .common_tagger import *
from .cells import Highway


log = get_logger(__name__)
MAX_WORD_LENGTH = 30


class CharacterTagger:
    """
    A class for character-based neural morphological tagger
    """
    def __init__(self, symbols: DefaultVocabulary, tags: DefaultVocabulary,
                 reverse=False, word_rnn="cnn",
                 char_embeddings_size=16, char_conv_layers=1,
                 char_window_size=5, char_filters=None,
                 char_filter_multiple=25, char_highway_layers=1,
                 conv_dropout=0.0, highway_dropout=0.0,
                 intermediate_dropout=0.0, lstm_dropout=0.0,
                 word_lstm_layers=1, word_lstm_units=128,
                 word_dropout=0.0, regularizer=None, verbose=1):
        self.symbols = symbols
        self.tags = tags
        self.reverse = reverse
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
        self.word_lstm_layers = word_lstm_layers
        self.word_lstm_units = word_lstm_units
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.regularizer = regularizer
        self.verbose = verbose
        self.initialize()
        log.info("{} symbols, {} tags in CharacterTagger".format(self.symbols_number_, self.tags_number_))
        self.build()

    def initialize(self):
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
        if self.regularizer is not None:
            self.regularizer = kreg.l2(self.regularizer)

    @property
    def symbols_number_(self):
        return len(self.symbols)

    @property
    def tags_number_(self):
        return len(self.tags)

    def build(self):
        word_inputs = kl.Input(shape=(None, MAX_WORD_LENGTH+2), dtype="int32")
        inputs = [word_inputs]
        word_outputs = self.build_word_cnn(word_inputs)
        outputs, lstm_outputs = self.build_basic_network(word_outputs)
        compile_args = {"optimizer": ko.nadam(lr=0.002, clipnorm=5.0),
                        "loss": "categorical_crossentropy", "metrics": ["accuracy"]}
        self.model_ = Model(inputs, outputs)
        self.model_.compile(**compile_args)
        if self.verbose > 0:
            log.info(str(self.model_.summary()))
        return self

    def build_word_cnn(self, inputs):
        # inputs = kl.Input(shape=(MAX_WORD_LENGTH,), dtype="int32")
        inputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number_},
                           output_shape=lambda x: tuple(x) + (self.symbols_number_,))(inputs)
        char_embeddings = kl.Dense(self.char_embeddings_size, use_bias=False)(inputs)
        conv_outputs = []
        self.char_output_dim_ = 0
        for window_size, filters_number in zip(self.char_window_size, self.char_filters):
            curr_output = char_embeddings
            curr_filters_number = (min(self.char_filter_multiple * window_size, 200)
                                   if filters_number is None else filters_number)
            for _ in range(self.char_conv_layers - 1):
                curr_output = kl.Conv2D(curr_filters_number, (1, window_size),
                                        padding="same", activation="relu",
                                        data_format="channels_last")(curr_output)
                if self.conv_dropout > 0.0:
                    curr_output = kl.Dropout(self.conv_dropout)(curr_output)
            curr_output = kl.Conv2D(curr_filters_number, (1, window_size),
                                    padding="same", activation="relu",
                                    data_format="channels_last")(curr_output)
            conv_outputs.append(curr_output)
            self.char_output_dim_ += curr_filters_number
        if len(conv_outputs) > 1:
            conv_output = kl.Concatenate(axis=-1)(conv_outputs)
        else:
            conv_output = conv_outputs[0]
        highway_input = kl.Lambda(kb.max, arguments={"axis": -2})(conv_output)
        if self.intermediate_dropout > 0.0:
            highway_input = kl.Dropout(self.intermediate_dropout)(highway_input)
        for i in range(self.char_highway_layers - 1):
            highway_input = Highway(activation="relu")(highway_input)
            if self.highway_dropout > 0.0:
                highway_input = kl.Dropout(self.highway_dropout)(highway_input)
        highway_output = Highway(activation="relu")(highway_input)
        return highway_output

    def build_basic_network(self, word_outputs):
        """
        Creates the basic network architecture,
        transforming word embeddings to intermediate outputs
        """
        if self.word_dropout > 0.0:
            lstm_outputs = kl.Dropout(self.word_dropout)(word_outputs)
        else:
            lstm_outputs = word_outputs
        for j in range(self.word_lstm_layers-1):
            lstm_outputs = kl.Bidirectional(
                kl.LSTM(self.word_lstm_units[j], return_sequences=True,
                        dropout=self.lstm_dropout))(lstm_outputs)
        lstm_outputs = kl.Bidirectional(
                kl.LSTM(self.word_lstm_units[-1], return_sequences=True,
                        dropout=self.lstm_dropout))(lstm_outputs)
        pre_outputs = kl.TimeDistributed(
                kl.Dense(self.tags_number_, activation="softmax",
                         activity_regularizer=self.regularizer),
                name="p")(lstm_outputs)
        return pre_outputs, lstm_outputs

    def _transform_batch(self, data, labels=None, transform_to_one_hot=True):
        L = max(len(x) for x in data)
        X = np.array([self._make_sent_vector(x, L) for x in data])
        if labels is not None:
            Y = np.array([self._make_tags_vector(y, L) for y in labels])
            if transform_to_one_hot:
                Y = to_one_hot(Y, len(self.tags))
            return X, Y
        else:
            return X

    def train_on_batch(self, data: List[List[str]], labels: List[List[str]], **kwargs):
        """
        Trains model on a single batch

        data: a batch of word sequences
        labels: a batch of correct tag sequences
        """
        X, Y = self._transform_batch(data, labels)
        # TO_DO: add weights to deal with padded instances
        return self.model_.train_on_batch(X, Y)

    def predict_on_batch(self, data: List[str], return_indexes=False):
        """
        Makes predictions on a single batch

        data: a batch of word sequences,
        -----------------------------------------
        answer: a batch of label sequences
        """
        X = self._transform_batch(data)
        Y = self.model_.predict_on_batch(X)
        labels = np.argmax(Y, axis=-1)
        answer: List[List[str]] = [None] * len(X)
        for i, elem in enumerate(labels):
            elem = elem[:len(data[i])]
            answer[i] = elem if return_indexes else self.tags.idxs2toks(elem)
        return answer

    def _make_sent_vector(self, sent, bucket_length=None):
        bucket_length = bucket_length or len(sent)
        answer = np.zeros(shape=(bucket_length, MAX_WORD_LENGTH+2), dtype=np.int32)
        for i, word in enumerate(sent):
            answer[i, 0] = self.tags.tok2idx("BEGIN")
            m = min(len(word), MAX_WORD_LENGTH)
            for j, x in enumerate(word[-m:]):
                answer[i, j+1] = self.symbols.tok2idx(x)
            answer[i, m+1] = self.tags.tok2idx("END")
            answer[i, m+2:] = self.tags.tok2idx("PAD")
        return answer

    def _make_tags_vector(self, tags, bucket_length=None):
        bucket_length = bucket_length or len(tags)
        answer = np.zeros(shape=(bucket_length,), dtype=np.int32)
        for i, tag in enumerate(tags):
            answer[i] = self.tags.tok2idx(tag)
        return answer

    def save(self, outfile):
        """
        outfile: file with model weights (other model components should be given in config)
        """
        self.model_.save_weights(outfile)

    def load(self, infile):
        self.model_.load_weights(infile)


