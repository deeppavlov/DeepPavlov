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

import os
import json
import copy

from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.registry import register_model

from keras.models import Model
from keras.layers import Dense, Input
import keras.metrics
import keras.optimizers


class KerasModel(Trainable, Inferable):
    def __init__(self, opt, *args, **kwargs):
        """
        Method initializes model using parameters from opt
        Args:
            opt: dictionary of parameters
            *args:
            **kwargs:
        """
        self.opt = copy.deepcopy(opt)
        if self.opt['model_from_saved'] == True:
            self.model = self.init_model_from_saved(model_name=self.opt['model_name'],
                                                    fname=self.opt['model_file'],
                                                    optimizer_name=self.opt['optimizer'],
                                                    lr=self.opt['lear_rate'],
                                                    decay=self.opt['lear_rate_decay'],
                                                    loss_name=self.opt['loss'],
                                                    metrics_names=self.opt['lear_metrics'])
        else:
            self.model = self.init_model_from_scratch(model_name=self.opt['model_name'],
                                                      optimizer_name=self.opt['optimizer'],
                                                      lr=self.opt['lear_rate'],
                                                      decay=self.opt['lear_rate_decay'],
                                                      loss_name=self.opt['loss'],
                                                      metrics_names=self.opt['lear_metrics'])

    def init_model_from_scratch(self, model_name, optimizer_name,
                                lr, decay, loss_name, metrics_names=None, loss_weights=None,
                                sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        """
        Method initializes model from scratch with given params
        Args:
            model_name:
            lr:
            decay:
            loss:
            metrics:

        Returns:

        """
        print('___Initializing model from scratch___')

        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(params=self.opt)
        else:
            raise AttributeError("Model %s is not defined" % model_name)

        optimizer_func = getattr(keras.optimizers, optimizer_name, None)
        if callable(optimizer_func):
            optimizer_ = optimizer_func(lr=lr, decay=decay)
        else:
            raise AttributeError("Optimizer %s is not callable" % optimizer_name)

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss %s is not defined" % loss_name)

        metrics_names = metrics_names.split(' ')
        metrics_funcs = []
        for i in range(len(metrics_names)):
            metrics_func = getattr(keras.metrics, metrics_names[i], None)
            if callable(metrics_func):
                metrics_funcs.append(metrics_func)
            else:
                raise AttributeError("Metric %s is not defined" % metrics_names[i])

        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics_funcs,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode,
                      weighted_metrics=weighted_metrics,
                      target_tensors=target_tensors)
        return model

    def init_model_from_saved(self, model_name, fname, optimizer_name,
                              lr, decay, loss_name, metrics_names=None, loss_weights=None,
                              sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        """
        Method initiliazes model from saved params and weights
        Args:
            model_name:
            fname:
            optimizer_name:
            lr:
            decay:
            loss:
            metrics:
            loss_weights:
            sample_weight_mode:
            weighted_metrics:
            target_tensors:

        Returns:

        """
        print('___Initializing model from saved___'
              '\nModel weights file is %s.h5'
              '\nNetwork parameters are from %s_opt.json' % (fname, fname))

        fname = self.opt.get('model_file', None) if fname is None else fname

        if os.path.isfile(fname + '_opt.json'):
            with open(fname + '_opt.json', 'r') as opt_file:
                self.opt = json.load(opt_file)
        else:
            raise IOError("Error: config file %s_opt.json of saved model does not exist" % fname)

        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(params=self.opt)
        else:
            raise AttributeError("Model %s is not defined" % model_name)

        model.load_weights(fname + '.h5')

        optimizer_func = getattr(keras.optimizers, optimizer_name, None)
        if callable(optimizer_func):
            optimizer_ = optimizer_func(lr=lr, decay=decay)
        else:
            raise AttributeError("Optimizer %s is not callable" % optimizer_name)

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss %s is not defined" % loss_name)

        metrics_names = metrics_names.split(' ')
        metrics_funcs = []
        for i in range(len(metrics_names)):
            metrics_func = getattr(keras.metrics, metrics_names[i], None)
            if callable(metrics_func):
                metrics_funcs.append(metrics_func)
            else:
                raise AttributeError("Metric %s is not defined" % metrics_names[i])

        model.compile(optimizer=optimizer_,
                      loss=loss,
                      metrics=metrics_funcs,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode,
                      weighted_metrics=weighted_metrics,
                      target_tensors=target_tensors)
        return model

    def train_on_batch(self, batch):
        """
        Method trains the model on a single batch of data
        Args:
            batch: tuple of (x,y) where x, y - lists of samples and their labels

        Returns: metrics values on a given batch

        """

        return self.model.train_on_batch(batch[0], batch[1])

    def train(self, data):
        """
        Method trains the model on a given data as a single batch
        Args:
            data: tuple of (x,y) where x, y - lists of samples and their labels

        Returns: metrics values on a given data

        """
        return self.train_on_batch(batch=data)

    def infer(self, batch, *args):
        """
        Method predicts on given batch
        Args:
            batch: tuple of (x,y) where x, y - lists of samples and their labels
            *args:

        Returns:

        """
        return self.model.predict_on_batch(batch)

    def save(self, fname=None):
        """
        Method saves the model parameters into <<fname>>_opt.json (or <<model_file>>_opt.json)
        and model weights into <<fname>>.h5 (or <<model_file>>.h5)
        Args:
            fname: file_path to save model. If not explicitly given seld.opt["model_file"] will be used

        Returns:

        """
        fname = self.opt.get('model_file', None) if fname is None else fname
        if fname:
            print("Saving model weights and params: " + fname + " ")
            self.model.save_weights(fname + '.h5')
            self.embedding_dict.save_items(fname)
            with open(fname + '_opt.json', 'w') as opt_file:
                json.dump(self.opt, opt_file)

    def MLP(self, opt):
        """
        Example of model function
        Build the un-compiled multilayer perceptron model
        Args:
            opt:

        Returns:

        """
        inp = Input(shape=opt['inp_shape'])
        output = inp
        for i in range(opt['n_layers']):
            output = Dense(opt['layer_size'], activation='relu')(output)
        output = Dense(self.n_classes, activation='softmax')(output)
        model = Model(inputs=inp, outputs=output)
        return model
