# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
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

import json
import os
import re
from logging import getLogger
from pathlib import Path
from typing import Union, List

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from overrides import overrides
from xeger import Xeger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel

log = getLogger(__name__)


@register("intent_catcher")
class IntentCatcher(NNModel):
    """Class for IntentCatcher Chainer's pipeline components."""

    def __init__(self, save_path: Union[str, Path], load_path: Union[str, Path],
        embeddings : str = 'use', limit : int = 10, multilabel : bool = False,
        number_of_layers : int = 0, number_of_intents : int = 1,
        hidden_dim : int = 256, mode : str = 'train', **kwargs) -> None:
        """Initializes IntentCatcher model.

        This model is mainly used for user intent detection in conversational systems.
        It provides some BERT-based embeddings for start and then fits a number
        of dense layers upon them for labels prediction.
        The main feature is that the user can provide regular expressions
        instead of actual phrases, and the model will derive phrases from it,
        thus making construction of the dataset easy and fast.
        The number of phrases generated from regexp is control by `limit` parameter.

        Args:
            save_path: Path to a directory with pretrained classifier and regexps for IntentCatcher.
            load_path: Path to a directory with pretrained classifier and regexps for IntentCatcher.
            embeddings: Input embeddings type. Provided embeddings are: USE and USE Large.
            limit: Maximum number of phrases, that are generated from input regexps.
            multilabel: Whether the task should be multilabel prediction or multiclass.
            number_of_layers: Number of hidden dense layers, that come after embeddings.
            number_of_intents: Number of output labels.
            hidden_dim: Dimension of hidden dense layers, that come after embeddings.
            mode: Train or infer mode. If infer - tries to load data from load_path.
            **kwargs: Additional parameters whose names will be logged but otherwise ignored.

        """
        super(IntentCatcher, self).__init__(save_path=save_path, load_path=load_path, **kwargs)
        if kwargs:
            log.info(f'{self.__class__.__name__} got additional init parameters {list(kwargs)} that will be ignored')
        urls = {
            'use':"https://tfhub.dev/google/universal-sentence-encoder/2",
            'use_large':"https://tfhub.dev/google/universal-sentence-encoder-large/2"
        }
        if embeddings not in urls:
            raise Exception(f"Provided embeddings type `{embeddings}` is not available. Available embeddings are: use, use_large.")
        self.limit = limit
        embedder = tfhub.Module(urls[embeddings])
        self.sentences = tf.placeholder(dtype=tf.string)
        self.embedded = embedder(self.sentences)
        mode = mode.lower().strip()
        if mode == 'infer':
            self.load()
        elif mode == 'train':
            log.info("Initializing NN")
            self.regexps = set()
            self.classifier = self._config_nn(number_of_layers, multilabel, hidden_dim, number_of_intents)
        else:
            raise Exception(f"Provided mode `{mode}` is not supported!")
        log.info("Configuring session")
        self.session = self._config_session()

    @staticmethod
    def _config_session():
        """
        Configure session for particular device

        Returns:
            tensorflow.Session
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.visible_device_list = '0'
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        return session

    def _config_nn(self, number_of_layers, multilabel, hidden_dim, number_of_intents) -> tf.keras.Model:
        """
        Initialize Neural Network upon embeddings.

        Returns:
            tf.keras.Model
        """
        if number_of_layers == 0:
            layers = [
                tf.keras.layers.Dense(
                    units=number_of_intents,
                    activation='softmax' if not multilabel else 'sigmoid'
                )
            ]
        elif number_of_layers > 0:
            layers = [
                tf.keras.layers.Dense(
                    units=hidden_dim,
                    activation='relu'
                )
            ]
            for i in range(number_of_layers-2):
                layers.append(
                    tf.keras.layers.Dense(
                        units=hidden_dim,
                        activation='relu'
                    )
                )
            layers.append(
                tf.keras.layers.Dense(
                    units=number_of_intents,
                    activation='softmax' if not multilabel else 'sigmoid'
                )
            )
        elif number_of_layers < 0:
            raise Exception("Number of layers should be >= 0")
        classifier = tf.keras.Sequential(layers=layers)
        classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if not multilabel else 'binary_crossentropy'
        )
        return classifier

    def train_on_batch(self, x: list, y: list) -> List[float]:
        """
        Train classifier on batch of data.

        Args:
            x: List of input sentences
            y: List of input encoded labels

        Returns:
            List[float]: list of losses.
        """
        assert len(x) == len(y), "Number of labels is not equal to the number of sentences"
        try:
            regexps = {(re.compile(s), l) for s, l in zip(x, y)}
        except Exception as e:
            log.error(f"Some sentences are not a consitent regular expressions")
            raise e
        xeger = Xeger(self.limit)
        self.regexps = self.regexps.union(regexps)
        generated_x = []
        generated_y = []
        for s, l in zip(x, y): # generate samples and add regexp
            gx = {xeger.xeger(s) for _ in range(self.limit)}
            generated_x.extend(gx)
            generated_y.extend([l for i in range(len(gx))])
        log.info(f"Original number of samples: {len(y)}, generated samples: {len(generated_y)}")
        embedded_x = self.session.run(self.embedded, feed_dict={self.sentences:generated_x}) # actual trainig
        loss = self.classifier.train_on_batch(embedded_x, generated_y)
        return loss

    def process_event(self, event_name, data):
        pass

    def __call__(self, x: List[str]) -> List[int]:
        """
        Predict probabilities.

        Args:
            x: list of input sentences.
        Returns:
            list of probabilities.
        """
        return self._predict_proba(x)

    def _predict_label(self, sentences: List[str]) -> List[int]:
        """
        Predict labels.

        Args:
            x: list of input sentences.
        Returns:
            list of labels.
        """
        labels = [None for i in range(len(sentences))]
        indx = []
        for i, s in enumerate(sentences):
            for reg, l in self.regexps:
                if reg.fullmatch(s):
                    labels[i] = l
            if not labels[i]:
                indx.append(i)
        sentences_to_nn = [sentences[i] for i in indx]
        x = self.session.run(self.embedded, feed_dict={self.sentences:sentences_to_nn})
        nn_predictions = self.classifier.predict_classes(x)
        for i, l in enumerate(nn_predictions):
            labels[indx[i]] = l
        return labels

    def _predict_proba(self, x: List[str]) -> List[float]:
        """
        Predict probabilities. Used in __call__.

        Args:
            x: list of input sentences.
        Returns:
            list of probabilities
        """
        x_embedded = self.session.run(self.embedded, feed_dict={self.sentences:x})
        probs = self.classifier.predict_proba(x_embedded)
        _, num_labels = probs.shape
        for i, s in enumerate(x):
            for reg, l in self.regexps:
                if reg.fullmatch(s):
                    probs[i] = np.zeros(num_labels)
                    probs[i, l] = 1.0
        return probs

    @overrides
    def save(self) -> None:
        """
        Save classifier parameters and regexps to self.save_path.
        """
        log.info("Saving model {} and regexps to {}".format(self.__class__.__name__, self.save_path))
        save_path = Path(self.save_path)
        if not save_path.exists():
            if save_path.parent.exists() and save_path.parent / "model" == save_path:
                os.mkdir(save_path.parent / "model")
        self.classifier.save(self.save_path / Path('nn.h5'))
        regexps = [{"regexp":reg.pattern, "label":str(l)} for reg, l in self.regexps]
        with open(self.save_path / Path('regexps.json'), 'w') as fp:
            json.dump(regexps, fp)

    @overrides
    def load(self) -> None:
        """
        Load classifier parameters and regexps from self.load_path.
        """
        log.info("Loading model {} and regexps from {}".format(self.__class__.__name__, self.save_path))
        self.classifier = tf.keras.models.load_model(self.load_path / Path("nn.h5"))
        with open(self.load_path / Path('regexps.json')) as fp:
            self.regexps = json.load(fp)
        self.regexps = [(re.compile(d['regexp']), int(d['label'])) for d in self.regexps]
