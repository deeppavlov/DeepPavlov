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

from abc import abstractmethod
from pathlib import Path
from copy import deepcopy, copy

import tensorflow as tf
import keras.metrics
import keras.optimizers
from overrides import overrides
from keras import backend as K
from keras.models import Model

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.file import save_json, read_json
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
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
        self.model = None
        self.epochs_done = 0
        self.batches_seen = 0
        self.train_examples_seen = 0

        super().__init__(save_path=kwargs.get('save_path', None),
                         load_path=kwargs.get('load_path', None),
                         url=kwargs.get('url', None),
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

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
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
