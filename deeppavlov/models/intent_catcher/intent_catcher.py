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
from typing import Union, Optional, List
from xeger import Xeger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.intent_catcher.utils import batch_samples

logger = getLogger(__name__)


@register("intent_catcher")
class IntentCatcher(Estimator, Serializable):
    """Class for IntentCatcher Chainer's pipeline components."""

    def __init__(self, save_path: Union[str, Path], load_path: Union[str, Path], catcher_params_path: Union[str, Path], **kwargs) -> None:
        """Initializes IntentCatcher on CPU (or GPU) and reads IntentCatcher modules params from yaml.

        Args:
            load_path: Path to a directory with pretrained classifier and regexps for IntentCatcher.
            catcher_params_path: Path to a file containig IntentCatcher modules params.

        """
        super(IntentCatcher, self).__init__(save_path=save_path, load_path=load_path, **kwargs)
        urls = {
            'use':"https://tfhub.dev/google/universal-sentence-encoder/2",
            'use_large':"https://tfhub.dev/google/universal-sentence-encoder-large/2"
        }
        self.regexps = []
        embedder = tfhub.Module(urls[catcher_params_path['embeddings']])
        self.sentences = tf.placeholder(dtype=tf.string)
        self.embedded = embedder(self.sentences)
        self.multulabel = catcher_params_path['multilabel']
        if config['number_of_layers'] == 0:
            layers = [
                tf.keras.layers.Dense(
                    units=config['number_of_intents'],
                    activation='softmax' if not config['multilabel'] else 'sigmoid',
                    input_shape=(None, int(embedded.shape[1]))
                )
            ]
        elif config['number_of_layers'] > 0:
            layers = [
                tf.keras.layers.Dense(
                    units=config['hidden_dim'],
                    activation='relu',
                    input_shape=(None, int(embedded.shape[1]))
                )
            ]
            for i in range(config['number_of_layers']-2):
                layers.append(
                    tf.keras.layers.Dense(
                        units=config['hidden_dim'],
                        activation='relu',
                        input_shape=(None, config['hidden_dim'])
                    )
                )
            layers.append(
                tf.keras.layers.Dense(
                    units=config['number_of_intents'],
                    activation='softmax' if not config['multilabel'] else 'sigmoid',
                    input_shape=(None, config['hidden_dim'])
                )
            )
        elif config['number_of_layers'] < 0:
            raise Exception("Number of layers should be >= 0")
        self.classifier = tf.keras.Sequential(layers=layers)
        self.classifier.compile(
            optimizer='adam',
            loss='categorical_crossentropy' if not config['multilabel'] else 'binary_crossentropy',
            metrics=config['metrics']
        )
        if 'device' in catcher_params_path:
            pass
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.tables_initializer())

    def fit(self, sentences: List[str], labels: List[Union[str, int]], limit=10, batch_size=64, epochs=1) -> None:
        """Train classifier"""
        xeger = Xeger(limit)
        assert len(sentences) == len(labels), logger.error("Number of labels is not equal to the number of sentences")
        intents = np.unique(labels)
        generated_sentences = []
        generated_labels = []
        logger.info("Generating samples from regexps")
        for s, l in zip(sentences, labels): # generate samples and add regexp
            try:
                self.regexps.append((re.compile(s), l))
            except Exception as e:
                logger.error(f"Sentence `{s}` is not a consitent regexp")
                raise e
            gs = {xeger.xeger(s) for _ in limit}
            generated_sentences.extend(gs)
            generated_labels.extend([l for i in range(len(gs))])
        logger.info("Training classifier")
        for epoch in range(epochs): # train classifier
            batch_generator = batch_samples(generated_sentences, generated_labels, batch_size)
            for x, y in batch_generator:
                x = self.session.run(self.embedded, feed_dict={self.sentences:x})
                self.classifier.train_on_batch(x, y)

    def __call__(self, sentences: List[str]) -> List[Union[str, int]]:
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

    def save(self) -> None:
        """Save classifier parameters and regexps"""
        logger.info("Saving model and regexps to {}".format(self.save_path))
        self.classifier.save(self.save_path / Path('nn.h5'))
        regexps = [{"regexp":reg.pattern, "label":str(l)} for reg, l in self.regexps]
        with open(self.save_path / Path('regexps.json')) as fp:
            json.dump(regexps, fp)

    def load(self) -> None:
        """Load classifier parameters and regexps"""
        logger.info("Loading model and regexps from {}".format(self.load_path))
        self.classifier = tf.keras.models.load_model(self.load_path / Path("model.h5"))
        with open(self.load_path / Path('regexps.json')) as fp:
            self.regexps = json.load(fp)
        self.regexps = [(re.compile(d['regexp']), d['label']) for d in self.regexps]
