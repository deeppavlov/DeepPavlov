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

from logging import getLogger
from pathlib import Path

import re
import json
import keras
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from xeger import Xeger
from sklearn.preprocessing import LabelEncoder

from typing import Union, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)


@register("intent_catcher")
class IntentCatcher(Estimator, Serializable):
    """Class for IntentCatcher Chainer's pipeline components."""

    def __init__(self, save_path: Union[str, Path], load_path: Union[str, Path],
        embeddings : str = 'use', limit : int = 10, multilabel : bool = False,
        number_of_layers : int = 0, number_of_intents : int = 1,
        hidden_dim : int = 256, **kwargs) -> None:
        """Initializes IntentCatcher on CPU (or GPU) and reads IntentCatcher modules params from yaml.

        Args:
            save_path: Path to a directory with pretrained classifier and regexps for IntentCatcher.
            load_path: Path to a directory with pretrained classifier and regexps for IntentCatcher.
            embeddings: Input embeddings type. Provided embeddings are: USE and USE Large.
            limit: Maximum number of phrases, that are generated from input regexps.
            multilabel: Whether the task should be multilabel prediction or multiclass.
            number_of_layers: Number of hidden dense layers, that come after embeddings.
            number_of_intents: Number of output labels.
            hidden_dim: Dimension of hidden dense layers, that come after embeddings.
            **kwargs: Additional parameters whose names will be logged but otherwise ignored.

        """
        super(IntentCatcher, self).__init__(save_path=save_path, load_path=load_path, **kwargs)
        if kwargs:
            log.info(f'{self.__class__.__name__} got additional init parameters {list(kwargs)} that will be ignored:')
        urls = {
            'use':"https://tfhub.dev/google/universal-sentence-encoder/2",
            'use_large':"https://tfhub.dev/google/universal-sentence-encoder-large/2"
        }
        if embeddings not in urls:
            raise Exception(f"Chosen embeddings {embeddings} are not available. Provided embeddings are: use, use_large.")
        self.limit = limit
        self.regexps = set()
        embedder = tfhub.Module(urls[embeddings])
        self.sentences = tf.placeholder(dtype=tf.string)
        self.embedded = embedder(self.sentences)
        self.multulabel =  multilabel
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
        self.classifier = tf.keras.Sequential(layers=layers)
        self.classifier.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if not catcher_config_params['multilabel'] else 'binary_crossentropy',
            metrics=catcher_config_params['metrics']
        )
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())

    def partial_fit(self, x: List[str], y: List[Union[str, int]]) -> None:
        """Train classifier on batch of data"""
        assert len(x) == len(y), logger.error("Number of labels is not equal to the number of sentences")
        try:
            regexps = {(re.compile(s), l) for s, l in zip(x, y)}
        except Exception as e:
            log.error(f"Some sentences are not a consitent regular expressions")
            raise e
        xeger = Xeger(limit)
        self.regexps.union(regexps)
        generated_x = []
        generated_y = []
        for s, l in zip(x, y): # generate samples and add regexp
            gx = {xeger.xeger(s) for _ in range(self.limit)}
            generated_x.extend(gx)
            generated_y.extend([l for i in range(len(gs))])
        log.info(f"Original number of samples: {len(y)}, generated samples: {len(generated_y)}")
        embedded_x = self.session.run(self.embedded, feed_dict={self.sentences:generated_x}) # actual trainig
        self.classifier.train_on_batch(embedded_x, generated_y)

    def __call__(self, sentences: List[str]) -> List[Union[str, int]]:
        """Predict probabilities"""
        return self.predict_proba(sentences)

    def predict_label(self, sentences: List[str]) -> List[Union[str, int]]:
        """Predict labels"""
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

    def predict_proba(self, sentences: List[str]) -> List[Union[str, int]]:
        """Predict probabilities"""
        x = self.session.run(self.embedded, feed_dict={self.sentences:sentences})
        probs = self.classifier.predict_proba(x)
        _, num_labels = probs.shape
        for i, s in enumerate(sentences):
            for reg, l in self.regexps:
                if reg.fullmatch(s):
                    probs[i] = np.zeros(num_labels)
                    probs[i, l] = 1.0
        return probs

    def save(self) -> None:
        """Save classifier parameters and regexps"""
        log.info("Saving model {} and regexps to {}".format(self.__class__.__name__, self.save_path))
        self.classifier.save(self.save_path / Path('nn.h5'))
        regexps = [{"regexp":reg.pattern, "label":str(l)} for reg, l in self.regexps]
        with open(self.save_path / Path('regexps.json'), 'w') as fp:
            json.dump(regexps, fp)

    def load(self) -> None:
        """Load classifier parameters and regexps"""
        log.info("Loading model {} and regexps from {}".format(self.__class__.__name__, self.save_path))
        self.classifier = tf.keras.models.load_model(self.load_path / Path("nn.h5"))
        with open(self.load_path / Path('regexps.json')) as fp:
            self.regexps = json.load(fp)
        self.regexps = [(re.compile(d['regexp']), int(d['label'])) for d in self.regexps]
