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
from typing import Dict
import numpy as np
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.models import Model
from keras.regularizers import l2

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.models.classifiers.intents import metrics as metrics_file
from deeppavlov.models.classifiers.intents.utils import labels2onehot, log_metrics, proba2labels
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.classifiers.intents.utils import md5_hashsum
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('intent_model')
class KerasIntentModel(KerasModel):
    """
    Class implements keras model for intent recognition task for multi-class multi-label data
    """
    def __init__(self,
                 opt: Dict,
                 embedder: FasttextEmbedder,
                 tokenizer: NLTKTokenizer,
                 classes=None,
                 vocabs=None,
                 **kwargs):
        """
        Initialize and train vocabularies, initializes embedder, tokenizer,
        and then initialize model using parameters from opt dictionary (from config),
        if model is being initialized from saved

        Args:
            vocabs: dictionary of considered vocabularies
            opt: model parameters for network and learning
            model_path: path to model serialization dir or file.
                            It is always an empty string and is ignored if it is not set in json config.
            model_dir: name of a serialization dir, can be default or set in json config
            model_file: name of a serialization file (usually binary model file),
                            can be default or set in json config
            embedder: instance of FasttextEmbedder class
            tokenizer: instance of NLTKTokenizer class
            **kwargs:
        """
        super().__init__(opt, **kwargs)

        # Tokenizer and vocabulary of classes
        self.tokenizer = tokenizer
        if classes:
            self.classes = np.sort(np.array(list(classes)))
        else:
            self.classes = np.sort(np.array(list(vocabs["classes_vocab"].keys())))
        self.n_classes = self.classes.shape[0]
        if self.n_classes == 0:
            ConfigError("Please, provide vocabulary with considered intents.")

        if 'add_metrics' in self.opt.keys():
            self.add_metrics = self.opt['add_metrics']
        else:
            self.add_metrics = None

        self.fasttext_model = embedder
        self.opt['embedding_size'] = self.fasttext_model.dim

        if self.fasttext_model.load_path:
            current_fasttext_md5 = md5_hashsum([self.fasttext_model.load_path])

        self.confident_threshold = self.opt['confident_threshold']

        # List of parameters that could be changed
        # when the model is initialized from saved and is going to be trained further
        changeable_params = {"lear_metrics": ["binary_accuracy"],
                             "confident_threshold": 0.5,
                             "optimizer": "Adam",
                             "lear_rate": 0.1,
                             "lear_rate_decay": 0.1,
                             "loss": "binary_crossentropy",
                             "coef_reg_cnn": 1e-4,
                             "coef_reg_den": 1e-4,
                             "dropout_rate": 0.5,
                             "epochs": 1,
                             "batch_size": 64,
                             "val_every_n_epochs": 1,
                             "verbose": True,
                             "val_patience": 5}

        # Reinitializing of parameters
        for param in changeable_params.keys():
            if param in opt.keys():
                self.opt[param] = opt[param]
            else:
                self.opt[param] = changeable_params[param]

        # Parameters required to init model
        params = {"model_name": self.opt['model_name'] if 'model_name' in self.opt.keys() else None,
                  "optimizer_name": self.opt['optimizer'],
                  "lr": self.opt['lear_rate'],
                  "decay": self.opt['lear_rate_decay'],
                  "loss_name": self.opt['loss'],
                  "metrics_names": self.opt['lear_metrics'],
                  "add_metrics_file": metrics_file}

        self.model = self.load(**params)

        # Reinitializing of parameters
        for param in changeable_params.keys():
            if param in opt.keys():
                self.opt[param] = opt[param]

        # Check if md5 hash sum of current loaded fasttext model
        # is equal to saved
        try:
            self.opt['fasttext_md5']
        except KeyError:
            self.opt['fasttext_md5'] = current_fasttext_md5
        else:
            if self.opt['fasttext_md5'] != current_fasttext_md5:
                raise ConfigError(
                    "Given fasttext model does NOT match fasttext model used previously to train loaded model")

        # Considered metrics including loss
        self.metrics_names = self.model.metrics_names

    def texts2vec(self, sentences):
        """
        Convert texts to vector representations using embedder and padding up to self.opt["text_size"] tokens
        Args:
            sentences: list of texts

        Returns:
            array of embedded texts
        """
        pad = np.zeros(self.opt['embedding_size'])

        embeddings_batch = self.fasttext_model([sen.split()[:self.opt['text_size']] for sen in sentences])
        embeddings_batch = [[pad] * (self.opt['text_size'] - len(tokens)) + tokens for tokens in embeddings_batch]

        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def train_on_batch(self, texts, labels):
        """
        Train the model on the given batch
        Args:
            batch - list of data where batch[0] is list of texts and batch[1] is list of labels

        Returns:
            loss and metrics values on the given batch
        """
        texts = self.tokenizer(list(texts))
        features = self.texts2vec(texts)
        onehot_labels = labels2onehot(labels, classes=self.classes)
        metrics_values = self.model.train_on_batch(features, onehot_labels)
        return metrics_values

    def infer_on_batch(self, batch, labels=None):
        """
        Infer the model on the given batch
        Args:
            batch - list of texts
            labels - list of labels

        Returns:
            loss and metrics values on the given batch, if labels are given
            predictions, otherwise
        """
        texts = self.tokenizer(batch)
        if labels:
            features = self.texts2vec(texts)
            onehot_labels = labels2onehot(labels, classes=self.classes)
            metrics_values = self.model.test_on_batch(features, onehot_labels)
            return metrics_values
        else:
            features = self.texts2vec(texts)
            predictions = self.model.predict(features)
            return predictions

    def __call__(self, data, predict_proba=False, *args):
        """
        Infer on the given data
        Args:
            data: [list of sentences]
            predict_proba: whether to return probabilities distribution or only labels-predictions
            *args:

        Returns:
            for each sentence:
                vector of probabilities to belong with each class
                or list of labels sentence belongs with
        """
        preds = np.array(self.infer_on_batch(data))

        if predict_proba:
            return preds
        else:
            return proba2labels(preds, confident_threshold=self.confident_threshold, classes=self.classes)

    def cnn_model(self, params):
        """
        Build un-compiled model of shallow-and-wide CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        outputs = []
        for i in range(len(params['kernel_sizes_cnn'])):
            output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(params['coef_reg_cnn']),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def dcnn_model(self, params):
        """
        Build un-compiled model of deep CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        if type(self.opt['filters_cnn']) is str:
            self.opt['filters_cnn'] = list(map(int, self.opt['filters_cnn'].split()))

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = inp

        for i in range(len(params['kernel_sizes_cnn'])):
            output = Conv1D(params['filters_cnn'][i], kernel_size=params['kernel_sizes_cnn'][i],
                            activation=None,
                            kernel_regularizer=l2(params['coef_reg_cnn']),
                            padding='same')(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D()(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def reset(self):
        pass
