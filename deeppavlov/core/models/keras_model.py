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

import inspect
from abc import abstractmethod
from copy import deepcopy

import tensorflow as tf
from keras import backend as K

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.tf_backend import TfModelMeta


log = get_logger(__name__)


class KerasModel(NNModel, metaclass=TfModelMeta):
    """
    Builds Keras model with TensorFlow backend.

    Attributes:
        epochs_done: number of epochs that were done
        batches_seen: number of epochs that were seen
        train_examples_seen: number of training samples that were seen
        sess: tf session
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize model using parameters from opt

        Args:
            kwargs (dict): Dictionary with model parameters
        """
        self.epochs_done = 0
        self.batches_seen = 0
        self.train_examples_seen = 0

        super().__init__(save_path=kwargs.get("save_path"),
                         load_path=kwargs.get("save_path"),
                         url=kwargs.get("url"),
                         mode=kwargs.get("mode"))

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

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

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
    """A wrapper over external Keras models. It is used, for example,
    to wrap :class:~deeppavlov.models.morpho_tagger.network.CharacterTagger.
    A subclass of :class:`~deeppavlov.core.models.nn_model.NNModel`

    Args:
        cls: the class to be wrapped
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