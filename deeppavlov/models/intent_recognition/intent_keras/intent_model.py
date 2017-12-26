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


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

import keras
import copy
import sys
import numpy as np
import fasttext
import re
import json
from pathlib import Path

from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.registry import register
from deeppavlov.models.embedders.fasttext_embedder import EmbeddingsDict
from deeppavlov.models.intent_recognition.intent_keras.utils import labels2onehot, log_metrics
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.core.common.file import save_json

from keras.models import Model
from keras.layers import Dense, Input, concatenate, Activation, Embedding
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import Bidirectional, LSTM
from keras.optimizers import Adam

import keras.metrics as keras_metrics_file
import keras.losses as keras_loss_file
from deeppavlov.core.common.attributes import check_attr_true
import deeppavlov.models.intent_recognition.intent_keras.metrics as metrics_file
from deeppavlov.core.common import paths
from keras import backend as K


@register('intent_model')
class KerasIntentModel(KerasModel):
    def __init__(self, opt, model_dir='keras_model', model_file='model', *args, **kwargs):

        self.opt = copy.deepcopy(opt)

        try:
            classes_file = self.opt['classes_file']
        except KeyError:  # if no classes path is passed in json
            try:
                classes_file = Path(paths.USR_PATH).joinpath('intents', 'classes.txt')
            except FileNotFoundError as e:
                raise (e,
                       "Something is bad with the path to dataset classes file."
                       "Provide the file path explicitly in json config.")

        with open(classes_file) as fin:
            self.classes = np.array(fin.read().split("\n"))

        self.n_classes = self.classes.shape[0]
        self.confident_threshold = self.opt['confident_threshold']
        if 'add_metrics' in self.opt.keys():
            self.add_metrics = self.opt['add_metrics'].split(' ')
            self.add_metrics_values = len(self.add_metrics) * [0.]
        else:
            self.add_metrics = None

        if self.opt['fasttext_model'] is not None:
            if Path(self.opt['fasttext_model']).is_file():
                self.embedding_dict = EmbeddingsDict(self.opt, self.opt['embedding_size'])
            else:
                raise IOError("Error: FastText model file does not exist")
        else:
            raise IOError("Error: FastText model file path is not given")

        if self.opt['model_from_saved']:
            self.model = self.load(model_name=self.opt['model_name'],
                                   fname=self.model_path_,
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

    def train_on_batch(self, batch):
        """
        Train the model on the given batch
        Args:
            batch - list of tuples (preprocessed text, labels)

        Returns:
            loss and metrics values on the given batch
        """
        texts = list(batch[0])
        labels = list(batch[1])
        self.embedding_dict.add_items(texts)
        features = self.texts2vec(texts)
        onehot_labels = labels2onehot(labels, classes=self.classes)
        metrics_values = self.model.train_on_batch(features, onehot_labels)
        return metrics_values

    @check_attr_true('train_now')
    def train(self, dataset, *args, **kwargs):
        """
        Method trains considered model using batches and validation
        Args:
            dataset: instance of class Dataset

        Returns: None

        """
        updates = 0
        val_loss = 1e100
        val_increase = 0
        epochs_done = 0

        n_train_samples = len(dataset.data['train'])

        valid_iter_all = dataset.iter_all(data_type='valid')
        valid_x = []
        valid_y = []
        for valid_i, valid_sample in enumerate(valid_iter_all):
            valid_x.append(valid_sample[0])
            valid_y.append(valid_sample[1])

        self.embedding_dict.add_items(valid_x)
        valid_x = self.texts2vec(valid_x)
        valid_y = labels2onehot(valid_y, classes=self.classes)

        print('\n____Training over %d samples____\n\n' % n_train_samples)

        while epochs_done < self.opt['epochs']:
            batch_gen = dataset.batch_generator(batch_size=self.opt['batch_size'],
                                                data_type='train')
            for step, batch in enumerate(batch_gen):
                metrics_values = self.train_on_batch(batch)
                updates += 1

                if self.opt['verbose'] and step % 50 == 0:
                    log_metrics(names=self.metrics_names,
                                values=metrics_values,
                                updates=updates,
                                mode='train')

            epochs_done += 1
            if epochs_done % self.opt['val_every_n_epochs'] == 0:
                if 'valid' in dataset.data.keys():
                    valid_metrics_values = self.model.test_on_batch(x=valid_x, y=valid_y)

                    log_metrics(names=self.metrics_names,
                                values=valid_metrics_values,
                                mode='valid')
                    if valid_metrics_values[0] > val_loss:
                        val_increase += 1
                        print("__Validation impatience %d out of %d" % (
                            val_increase, self.opt['val_patience']))
                        if val_increase == self.opt['val_patience']:
                            print("___Stop training: validation is out of patience___")
                            break
                    val_loss = valid_metrics_values[0]
            print('epochs_done: %d' % epochs_done)

        self.save()

    def infer(self, data, *args):
        """
        Return predictions on the given data (batch of texts or one text)
        """
        if type(data) is str:
            self.embedding_dict.add_items([data])
            features = self.texts2vec([data])
            preds = self.model.predict_on_batch(features)[0]
        else:
            self.embedding_dict.add_items(data)
            features = self.texts2vec(data)
            preds = self.model.predict_on_batch(features)
        return preds

    def save(self, fname=None):
        fname = self.model_path_.name if fname is None else fname
        opt_fname = fname + '_opt.json'
        weights_fname = fname + '.h5'

        opt_path = Path.joinpath(self.model_path_, opt_fname)
        weights_path = Path.joinpath(self.model_path_, weights_fname)
        emb_path = Path.joinpath(self.model_path_, fname)
        print("[ saving model: {} ]".format(str(opt_path)))
        self.model.save_weights(weights_path)
        self.embedding_dict.save_items(str(emb_path))

        save_json(self.opt, opt_path)

        return True

    def cnn_model(self, params):
        """
        Build the uncompiled model of shallow-and-wide CNN
        :return: model
        """
        if type(self.opt['kernel_sizes_cnn']) is str:
            self.opt['kernel_sizes_cnn'] = [int(x) for x in
                                            self.opt['kernel_sizes_cnn'].split(' ')]

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
        Build the uncompiled model of deep CNN
        :return: model
        """
        if type(self.opt['kernel_sizes_cnn']) is str:
            self.opt['kernel_sizes_cnn'] = [int(x) for x in
                                            self.opt['kernel_sizes_cnn'].split(' ')]

        if type(self.opt['filters_cnn']) is str:
            self.opt['filters_cnn'] = [int(x) for x in
                                       self.opt['filters_cnn'].split(' ')]

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
