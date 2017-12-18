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

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

from abc import abstractmethod
from pathlib import Path

import keras
import copy
import os
import numpy as np
import fasttext
import re
import json

from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.registry import register_model
from deeppavlov.core.models.keras_model import KerasModel

from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation, Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Bidirectional, LSTM
from keras.optimizers import Adam

import intent_recognition.metrics as metrics_file
from utils import EmbeddingsDict



@register_model('intent_model_from_parent')
class KerasIntentModelFromParent(KerasModel):
    def __init__(self, opt, classes, *args, **kwargs):
        #super().__init__(opt)
        self.opt = copy.deepcopy(opt)
        self.classes = classes
        self.n_classes = self.classes.shape[0]
        self.confident_threshold = self.opt['confident_threshold']
        if 'add_metrics' in self.opt.keys():
            self.add_metrics = self.opt['add_metrics'].split(' ')
            self.add_metrics_values = len(self.add_metrics) * [0.]
        else:
            self.add_metrics = None

        self.opt['kernel_sizes_cnn'] = [int(x) for x in
                                        self.opt['kernel_sizes_cnn'].split(' ')]
        print(self.opt)

        if self.opt['model_from_saved'] == True:
            self.model = self.load(model_name=self.opt['model_name'],
                                   fname=self.opt['model_file'],
                                   optimizer_name=self.opt['optimizer'],
                                   lr=self.opt['lear_rate'],
                                   decay=self.opt['lear_rate_decay'],
                                   loss_name=self.opt['loss'],
                                   metrics_names=self.opt['lear_metrics'],
                                   add_metrics_file=metrics_file)
        else:
            self.model = self.init_model_from_scratch(model_name=self.opt['model_name'],
                                                      optimizer_name=self.opt['optimizer'],
                                                      lr=self.opt['lear_rate'],
                                                      decay=self.opt['lear_rate_decay'],
                                                      loss_name=self.opt['loss'],
                                                      metrics_names=self.opt['lear_metrics'],
                                                      add_metrics_file=metrics_file)

        self.metrics_names = self.model.metrics_names
        self.metrics_values = len(self.metrics_names) * [0.]

        if self.opt['fasttext_model'] is not None:
            if os.path.isfile(self.opt['fasttext_model']):
                self.embedding_dict = EmbeddingsDict(self.opt, self.opt['embedding_size'])
            else:
                raise IOError("Error: FastText model file does not exist")
        else:
            raise IOError("Error: FastText model file path is not given")

    def texts2vec(self, sentences):
        embeddings_batch = []
        for sen in sentences:
            embeddings = []
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            if len(tokens) > self.opt['text_size']:
                tokens = tokens[:self.opt['text_size']]
            for tok in tokens:
                embeddings.append(self.embedding_dict.tok2emb.get(tok))
            if len(tokens) < self.opt['text_size']:
                pads = [np.zeros(self.opt['embedding_size'])
                        for _ in range(self.opt['text_size'] - len(tokens))]
                embeddings = pads + embeddings

            embeddings = np.asarray(embeddings)
            embeddings_batch.append(embeddings)

        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def labels2onehot(self, labels):
        eye = np.eye(self.n_classes)
        y = []
        for sample in labels:
            curr = np.zeros(self.n_classes)
            for intent in sample:
                if intent not in self.classes:
                    print('Warning: unknown intent %s detected' % intent)
                    curr += eye[np.where(self.classes == 'unknown')[0]].reshape(-1)
                else:
                    curr += eye[np.where(self.classes == intent)[0]].reshape(-1)
            y.append(curr)
        y = np.asarray(y)
        return y

    def proba2labels(self, proba):
        y = []
        for sample in proba:
            to_add = np.where(sample > self.confident_threshold)[0]
            if len(to_add) > 0:
                y.append(self.classes[to_add])
            else:
                y.append([self.classes[np.argmax(sample)]])
        y = np.asarray(y)
        return y

    def proba2onehot(self, proba):
        return self.labels2onehot(self.proba2labels(proba))

    def train_on_batch(self, batch):
        """
        Train the model
        Args:
            batch - list of tuples (preprocessed text, labels)
        """
        texts = list(batch[0])
        labels = list(batch[1])
        self.embedding_dict.add_items(texts)
        features = self.texts2vec(texts)
        onehot_labels = self.labels2onehot(labels)
        self.metrics_values = self.model.train_on_batch(features, onehot_labels)
        if self.add_metrics is not None:
            preds = self.model.predict_on_batch(features)
            for i, add_metrics in enumerate(self.add_metrics):
                self.add_metrics_values[i] = add_metrics(onehot_labels, preds)
        return True

    def train(self, data):
        return self.train_on_batch(batch=data)

    def infer(self, batch, *args):
        """
        Return prediction.
        """
        self.embedding_dict.add_items(batch)
        features = self.texts2vec(batch)
        preds = self.model.predict_on_batch(features)
        return preds

    def cnn_model(self, params):
        """
        Build the incompiled model
        :return: model
        """
        inp = Input(shape=(params['text_size'], params['embedding_size']))
        print(params)
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

    def bilstm_model(self, params):
        """
        Build the incompiled model
        :return: model
        """
        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate_lstm']))(inp)

        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['n_classes'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model
