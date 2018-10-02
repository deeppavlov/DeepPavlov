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
import copy
import inspect

from abc import abstractmethod
from pathlib import Path
from copy import deepcopy, copy

import tensorflow as tf
import keras.metrics
import keras.optimizers
from typing import Dict
from overrides import overrides
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.file import save_json, read_json
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.tf_backend import TfModelMeta
from .tf_backend import TfModelMeta


log = get_logger(__name__)


class KerasModel(NNModel, metaclass=TfModelMeta):
    """
    Builds Keras model with TensorFlow backend.

    Attributes:
        opt: dictionary with all model parameters
        model: keras model itself
        epochs_done: number of epochs that were done
        batches_seen: number of epochs that were seen
        train_examples_seen: number of training samples that were seen
        sess: tf session
        optimizer: keras.optimizers instance
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize model using parameters from opt

        Args:
            kwargs (dict): Dictionary with model parameters
        """
        self.opt = copy(kwargs)
        save_path = self.opt.get('save_path', None)
        load_path = self.opt.get('load_path', None)
        url = self.opt.get('url', None)
        self.model = None
        self.epochs_done = 0
        self.batches_seen = 0
        self.train_examples_seen = 0

        super().__init__(save_path=save_path,
                         load_path=load_path,
                         url=url,
                         mode=kwargs['mode'])

    @staticmethod
    def _config_session():
        """
        Configure session for particular device

        Returns:
            tensorflow.Session
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '0'
        return tf.Session(config=config)

    def init_model_from_scratch(self, model_name: str):
        """
        Initialize uncompiled model from scratch with given params

        Args:
            model_name: name of model function described as a method of this class

        Returns:
            compiled model with given network and learning parameters
        """
        log.info(f'[initializing `{self.__class__.__name__}` from scratch as {model_name}]')
        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(**self.opt)
        else:
            raise AttributeError("Model {} is not defined".format(model_name))

        return model

    @overrides
    def load(self, model_name: str):
        """
        Initialize uncompiled model from saved params and weights

        Args:
            model_name: name of model function described as a method of this class

        Returns:
            model with loaded weights and network parameters from files
            but compiled with given learning parameters
        """
        if self.load_path:
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            opt_path = Path("{}_opt.json".format(str(self.load_path.resolve())))
            weights_path = Path("{}.h5".format(str(self.load_path.resolve())))

            if opt_path.exists() and weights_path.exists():

                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))

                self.opt = read_json(opt_path)

                model_func = getattr(self, model_name, None)
                if callable(model_func):
                    model = model_func(**self.opt)
                else:
                    raise AttributeError("Model {} is not defined".format(model_name))

                log.info("[loading weights from {}]".format(weights_path.name))
                model.load_weights(str(weights_path))

                return model
            else:
                return self.init_model_from_scratch(model_name)
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            return self.init_model_from_scratch(model_name)

    def compile(self, model: Model, optimizer_name: str, loss_name: str,
                lear_rate: float = 0.01, lear_rate_decay: float = 0.):
        """
        Compile model with given optimizer and loss

        Args:
            model: keras uncompiled model
            optimizer_name: name of optimizer from keras.optimizers
            loss_name: loss function name (from keras.losses)
            lear_rate: learning rate.
            lear_rate_decay: learning rate decay.

        Returns:

        """
        optimizer_func = getattr(keras.optimizers, optimizer_name, None)
        if callable(optimizer_func):
            if not (lear_rate is None):
                if not (lear_rate_decay is None):
                    self.optimizer = optimizer_func(lr=lear_rate, decay=lear_rate_decay)
                else:
                    self.optimizer = optimizer_func(lr=lear_rate)
            elif not (lear_rate_decay is None):
                self.optimizer = optimizer_func(decay=lear_rate_decay)
            else:
                self.optimizer = optimizer_func()
        else:
            raise AttributeError("Optimizer {} is not defined in `keras.optimizers`".format(optimizer_name))

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(loss_name))

        model.compile(optimizer=self.optimizer,
                      loss=loss)
        return model

    @overrides
    def save(self, fname: str = None) -> None:
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

        # if model was loaded from one path and saved to another one
        # then change load_path to save_path for config
        self.opt["epochs_done"] = self.epochs_done
        self.opt["final_lear_rate"] = K.eval(self.optimizer.lr) / (1. +
                                                                   K.eval(self.optimizer.decay) * self.batches_seen)

        if self.opt.get("load_path") and self.opt.get("save_path"):
            if self.opt.get("save_path") != self.opt.get("load_path"):
                self.opt["load_path"] = str(self.opt["save_path"])
        save_json(self.opt, opt_path)

    @abstractmethod
    def reset(self):
        pass

    def process_event(self, event_name: str, data: dict):
        """
        Process event after epoch
        Args:
            event_name: whether event is send after epoch or batch.
                    Set of values: ``"after_epoch", "after_batch"``
            data: event data (dictionary)

        Returns:
            None
        """
        if event_name == "after_epoch":
            self.epochs_done = data["epochs_done"]
            self.batches_seen = data["batches_seen"]
            self.train_examples_seen = data["train_examples_seen"]
        return


class ExternalKerasWrapper(NNModel, metaclass=TfModelMeta):
    """A wrapper over morphological tagger, implemented in
    :class:~deeppavlov.models.morpho_tagger.network.CharacterTagger.
    A subclass of :class:`~deeppavlov.core.models.nn_model.NNModel`

    Args:
        save_path: the path where model is saved
        load_path: the path from where model is loaded
        mode: usage mode
        **kwargs: a dictionary containing model parameters specified in the main part
            of json config that corresponds to the model
    """
    def __init__(self, cls: type, save_path: str = None, load_path: str = None, mode: str = None, **kwargs):
        # Calls parent constructor. Results in creation of save_folder if it doesn't exist
        super().__init__(save_path=save_path, load_path=load_path, mode=mode)

        # Dicts are mutable! To prevent changes in config dict outside this class
        # we use deepcopy
        opt = deepcopy(kwargs)

        # Finds all input parameters of the network __init__ to pass them into network later
        network_parameter_names = list(inspect.signature(cls.__init__).parameters)
        # Fills all provided parameters from opt (opt is a dictionary formed from the model
        # json config file, except the "name" field)
        network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self._net = cls(**network_parameters)

        # Finds all parameters for network train to pass them into train method later
        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)

        # Fills all provided parameters from opt
        train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}
        self.train_parameters = train_parameters
        self.opt = opt

        # Tries to load the model from model `load_path`, if it is available
        self.load()

    @staticmethod
    def _config_session():
        """
        Configure session for particular device

        Returns:
            tensorflow.Session
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '0'
        return tf.Session(config=config)

    def load(self):
        """Checks existence of the model file, loads the model if the file exists"""

        # General way (load path from config assumed to be the path
        # to the file including extension of the file model)
        model_file_exist = self.load_path.exists()
        path = str(self.load_path.resolve())
        # Check presence of the model files
        if model_file_exist:
            log.info('[loading model from {}]'.format(path))
            self._net.load(path)

    def save(self):
        """Saves model to the save_path, provided in config. The directory is
        already created by super().__init__, which is called in __init__ of this class"""
        path = str(self.save_path.absolute())
        log.info('[saving model to {}]'.format(path))
        self._net.save(path)

    def train_on_batch(self, *args):
        """Trains the model on a single batch.

        Args:
            *args: the list of network inputs.
            Last element of `args` is the batch of targets,
            all previous elements are training data batches
        """
        *data, labels = args
        self._net.train_on_batch(data, labels)

    def __call__(self, *x_batch, **kwargs):
        """
        Predicts answers on batch elements.

        Args:
            instance: a batch to predict answers on
        """
        with self.graph.as_default():
            K.set_session(self.sess)
            return self._net.predict_on_batch(x_batch, **kwargs)