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

# from deeppavlov.common import paths
# from deeppavlov.models.trainable import Trainable
# from deeppavlov.models.inferable import Inferable

from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation, Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Bidirectional, LSTM
from keras.optimizers import Adam

from utils import EmbeddingsDict

# class KerasIntentModel(Trainable, Inferable):
class KerasIntentModel(object):

    def __init__(self, opt, classes):
        # super.__init__()
        # self._model_dir_path = ''
        # self._model_fpath = ''
        # self._model_path = Path(paths.USR_PATH).joinpath(self._model_dir_path, self._model_fpath)
        self.opt = copy.deepcopy(opt)
        self.opt['kernel_sizes_cnn'] = [int(x) for x in opt['kernel_sizes_cnn'].split(' ')]
        self.opt['lear_metrics_list'] = opt['lear_metrics'].split(' ')
        self.classes = classes
        self.n_classes = self.classes.shape[0]
        self.confident_threshold = self.opt['confident_threshold']
        if 'add_metrics' in self.opt.keys():
            self.add_metrics = self.opt['add_metrics'].split(' ')
            self.add_metrics_values = len(self.add_metrics) * [0.]
        else:
            self.add_metrics = None

        self.network_params = dict()
        self.learning_params = dict()
        self.get_params()

        if self.opt['model_name'] == 'cnn_model':
            model_function = self.cnn_model()
        elif self.opt['model_name'] == 'bilstm_model':
            model_function = self.bilstm_model()

        if self.opt['model_from_saved'] == False:
            self.model = self.init_model_from_scratch(model_function=model_function,
                                                      lr=self.learning_params['lear_rate'],
                                                      decay=self.learning_params['lear_rate_decay'],
                                                      loss=self.learning_params['loss'],
                                                      metrics=self.learning_params['lear_metrics_list'])
        else:
            self.model = self.init_model_from_saved(model_function=model_function,
                                                    lr=self.learning_params['lear_rate'],
                                                    decay=self.learning_params['lear_rate_decay'],
                                                    loss=self.learning_params['loss'],
                                                    metrics=self.learning_params['lear_metrics_list'])

        self.metrics_names = self.model.metrics_names
        self.metrics_values = len(self.metrics_names) * [0.]

        if self.opt['fasttext_model'] is not None:
            if os.path.isfile(self.opt['fasttext_model']):
                self.embedding_dict = EmbeddingsDict(self.opt, self.opt['embedding_size'])
            else:
                raise IOError("Error: FastText model file does not exist")
        else:
            raise IOError("Error: FastText model file path is not given")

    def get_params(self):
        list_network_params = ['text_size', 'embedding_size', 'n_classes', 'coef_reg_den', 'dense_size',
                               'kernel_sizes_cnn', 'filters_cnn', 'coef_reg_cnn',
                               'units_lstm', 'coef_reg_lstm', 'dropout_rate_lstm']
        list_learning_params = ['dropout_rate', 'lear_rate', 'lear_rate_decay',
                                'lear_metrics_list', 'loss',
                                'batch_size', 'epochs'] # TODO: move it to training
        for param in list_network_params:
            if param in self.opt.keys():
                self.network_params[param] = self.opt[param]
        for param in list_learning_params:
            if param in self.opt.keys():
                self.learning_params[param] = self.opt[param]
        return True

    def init_model_from_scratch(self, model_function, lr, decay,
                                loss, metrics=None):
        model = model_function
        optimizer_ = Adam(lr=lr, decay=decay)
        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics)
        print('___Initializing model from scratch___')
        return model

    def init_model_from_saved(self, model_function, fname, lr, decay,
                              loss, metrics=None, loss_weights=None,
                              sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        print('___Initializing model from saved___'
              '\nModel weights file is %s.h5'
              '\nNetwork parameters are from %s_opt.json' % (fname, fname))
        if os.path.isfile(fname + '_opt.json'):
            with open(fname + '_opt.json', 'r') as opt_file:
                # TODO: network params from json, learning params from current config
                old_opt = json.load(opt_file)
                for param in self.network_params.keys():
                    self.network_params[param] = old_opt[param]
        else:
            raise IOError("Error: config file %s_opt.json of saved model does not exist" % fname)
        model = model_function
        model.load_weights(fname + '.h5')
        optimizer_ = Adam(lr=lr, decay=decay)
        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode,
                      weighted_metrics=weighted_metrics,
                      target_tensors=target_tensors)
        return model

    def texts2vec(self, sentences):
        embeddings_batch = []
        for sen in sentences:
            embeddings = []
            tokens = sen.split(' ')
            tokens = [el for el in tokens if el != '']
            if len(tokens) > self.network_params['text_size']:
                tokens = tokens[:self.network_params['text_size']]
            for tok in tokens:
                embeddings.append(self.embedding_dict.tok2emb.get(tok))
            if len(tokens) < self.network_params['text_size']:
                pads = [np.zeros(self.network_params['embedding_size'])
                        for _ in range(self.network_params['text_size'] - len(tokens))]
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

    def infer(self, batch, *args):
        """
        Return prediction.
        """

        features = self.texts2vec(batch)
        preds = self.model.predict_on_batch(features)
        return preds

    def save(self, fname):
        # TODO: model_file is in opt??
        fname = self.opt.get('model_file', None) if fname is None else fname

        if fname:
            print("[ saving model: " + fname + " ]")
            self.model.save_weights(fname + '.h5')
            self.embedding_dict.save_items(fname)

            with open(fname + '_opt.json', 'w') as opt_file:
                json.dump(self.opt, opt_file)
        return True

    def cnn_model(self):
        """
        Build the incompiled model
        :return: model
        """
        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))

        outputs = []
        for i in range(len(self.opt['kernel_sizes_cnn'])):
            output_i = Conv1D(self.opt['filters_cnn'], kernel_size=self.opt['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(self.opt['coef_reg_cnn']),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.opt['dense_size'], activation=None,
                       kernel_regularizer=l2(self.opt['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(self.opt['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_model(self):
        """
        Build the incompiled model
        :return: model
        """
        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))

        output = Bidirectional(LSTM(self.opt['units_lstm'], activation='tanh',
                                    kernel_regularizer=l2(self.opt['coef_reg_lstm']),
                                    dropout=self.opt['dropout_rate_lstm']))(inp)

        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.opt['dense_size'], activation=None,
                       kernel_regularizer=l2(self.opt['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=self.opt['dropout_rate'])(output)
        output = Dense(self.opt['n_classes'], activation=None,
                       kernel_regularizer=l2(self.opt['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model
