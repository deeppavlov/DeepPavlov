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

from abc import abstractmethod
from pathlib import Path
from copy import deepcopy, copy

import tensorflow as tf
import keras.metrics
import keras.optimizers
from typing import Dict
from overrides import overrides
from .tf_backend import TfModelMeta
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.file import save_json, read_json
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


class KerasModel(NNModel, metaclass=TfModelMeta):
    """
    Class builds keras model with tensorflow backend
    """

    def __init__(self, **kwargs):
        """
        Initialize model using parameters from opt
        Args:
            opt: model parameters
            *args:
            **kwargs:
        """
        self.opt = copy(kwargs)
        save_path = self.opt.get('save_path', None)
        load_path = self.opt.get('load_path', None)
        url = self.opt.get('url', None)
        self.model = None

        super().__init__(save_path=save_path,
                         load_path=load_path,
                         url=url,
                         mode=kwargs['mode'])

        self.sess = self._config_session()
        K.set_session(self.sess)

    def _config_session(self):
        """
        Configure session for particular device
        Returns:
            tensorflow.Session
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '0'
        return tf.Session(config=config)

    def init_model_from_scratch(self, model_name, optimizer_name, loss_name, lear_rate=None, lear_rate_decay=None):
        """
        Initialize model from scratch with given params
        Args:
            model_name: name of model function described as a method of this class
            optimizer_name: name of optimizer from keras.optimizers
            lr: learning rate
            decay: learning rate decay
            loss_name: loss function name (from keras.losses)

        Returns:
            compiled model with given network and learning parameters
        """
        log.info("[initializing `{}` from scratch]".format(self.__class__.__name__))
        print(model_name)
        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(params=self.opt)
        else:
            raise AttributeError("Model {} is not defined".format(model_name))

        optimizer_func = getattr(keras.optimizers, optimizer_name, None)
        if callable(optimizer_func):
            if not(lear_rate is None):
                if not(lear_rate_decay is None):
                    optimizer_ = optimizer_func(lr=lear_rate, decay=lear_rate_decay)
                else:
                    optimizer_ = optimizer_func(lr=lear_rate)
            elif not(lear_rate_decay is None):
                optimizer_ = optimizer_func(decay=lear_rate_decay)
            else:
                optimizer_ = optimizer_func()
        else:
            raise AttributeError("Optimizer {} is not defined in `keras.optimizers`".format(optimizer_name))

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined in `keras.losses`".format(loss_name))

        model.compile(optimizer=optimizer_, loss=loss)
        return model

    @overrides
    def load(self, model_name, optimizer_name, loss_name, lear_rate=None, lear_rate_decay=None):
        """
        Initialize model from saved params and weights
        Args:
            model_name: name of model function described as a method of this class
            optimizer_name: name of optimizer from keras.optimizers
            lr: learning rate
            decay: learning rate decay
            loss_name: loss function name (from keras.losses)

        Returns:
            model with loaded weights and network parameters from files
            but compiled with given learning parameters
        """
        if self.load_path:
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided save path is incorrect!")

            opt_path = Path("{}_opt.json".format(str(self.load_path.resolve())))
            weights_path = Path("{}.h5".format(str(self.load_path.resolve())))

            if opt_path.exists() and weights_path.exists():

                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))

                self.opt = read_json(opt_path)

                model_func = getattr(self, model_name, None)
                if callable(model_func):
                    model = model_func(params=self.opt)
                else:
                    raise AttributeError("Model {} is not defined".format(model_name))

                log.info("[loading weights from {}]".format(weights_path.name))
                model.load_weights(str(weights_path))

                optimizer_func = getattr(keras.optimizers, optimizer_name, None)
                if callable(optimizer_func):
                    if not (lear_rate is None):
                        if not (lear_rate_decay is None):
                            optimizer_ = optimizer_func(lr=lear_rate, decay=lear_rate_decay)
                        else:
                            optimizer_ = optimizer_func(lr=lear_rate)
                    elif not (lear_rate_decay is None):
                        optimizer_ = optimizer_func(decay=lear_rate_decay)
                    else:
                        optimizer_ = optimizer_func()
                else:
                    raise AttributeError("Optimizer {} is not defined in `keras.optimizers`".format(optimizer_name))

                loss_func = getattr(keras.losses, loss_name, None)
                if callable(loss_func):
                    loss = loss_func
                else:
                    raise AttributeError("Loss {} is not defined".format(loss_name))

                model.compile(optimizer=optimizer_,
                              loss=loss)
                return model
            else:
                return self.init_model_from_scratch(model_name, optimizer_name, loss_name, lear_rate, lear_rate_decay)
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            return self.init_model_from_scratch(model_name, optimizer_name, loss_name, lear_rate, lear_rate_decay)

    @overrides
    def save(self, fname=None):
        """
        Save the model parameters into <<fname>>_opt.json (or <<ser_file>>_opt.json)
        and model weights into <<fname>>.h5 (or <<ser_file>>.h5)
        Args:
            fname: file_path to save model. If not explicitly given seld.opt["ser_file"] will be used

        Returns:
            None
        """
        if not fname:
            fname = self.save_path
        else:
            fname = Path(fname).resolve()

        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")
        else:
            opt_path = f"{fname}_opt.json"
            weights_path = f"{fname}.h5"
            log.info(f"[saving model to {opt_path}]")
            self.model.save_weights(weights_path)


        save_json(self.opt, opt_path)
        return True

    def mlp(self, opt):
        """
        Example of model function
        Build un-compiled multilayer perceptron model
        Args:
            opt: dictionary of parameters

        Returns:
            un-compiled Keras model
        """
        inp = Input(shape=opt['inp_shape'])
        output = inp
        for i in range(opt['n_layers']):
            output = Dense(opt['layer_size'], activation='relu')(output)
        output = Dense(1, activation='softmax')(output)

        model = Model(inputs=inp, outputs=output)
        return model

    @abstractmethod
    def reset(self):
        pass
