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
from logging import getLogger

import tensorflow.compat.v1 as tf
from tensorflow.keras import backend as K
from overrides import overrides

from deeppavlov.core.models.lr_scheduled_model import LRScheduledModel
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.models.tf_backend import TfModelMeta

log = getLogger(__name__)


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
        Initialize model using keyword parameters

        Args:
            kwargs: Dictionary with model parameters
        """
        self.epochs_done = 0
        self.batches_seen = 0
        self.train_examples_seen = 0

        super().__init__(save_path=kwargs.get("save_path"),
                         load_path=kwargs.get("load_path"),
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
    def load(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        pass

    def process_event(self, event_name: str, data: dict) -> None:
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


class LRScheduledKerasModel(LRScheduledModel, KerasModel):
    """
    KerasModel enhanced with optimizer, learning rate and momentum
    management and search.
    """

    def __init__(self, **kwargs):
        """
        Initialize model with given parameters

        Args:
            **kwargs: dictionary of parameters
        """
        self.opt = kwargs
        KerasModel.__init__(self, **kwargs)
        if not(isinstance(kwargs.get("learning_rate"), float) and isinstance(kwargs.get("learning_rate_decay"), float)):
            LRScheduledModel.__init__(self, **kwargs)

    @abstractmethod
    def get_optimizer(self):
        """
        Return an instance of keras optimizer
        """
        pass

    @overrides
    def _init_learning_rate_variable(self):
        """
        Initialize learning rate

        Returns:
            None
        """
        return None

    @overrides
    def _init_momentum_variable(self):
        """
        Initialize momentum

        Returns:
            None
        """
        return None

    @overrides
    def get_learning_rate_variable(self):
        """
        Extract value of learning rate from optimizer

        Returns:
            learning rate value
        """
        return self.get_optimizer().lr

    @overrides
    def get_momentum_variable(self):
        """
        Extract values of momentum variables from optimizer

        Returns:
            optimizer's `rho` or `beta_1`
        """
        optimizer = self.get_optimizer()
        if hasattr(optimizer, 'rho'):
            return optimizer.rho
        elif hasattr(optimizer, 'beta_1'):
            return optimizer.beta_1
        return None

    @overrides
    def _update_graph_variables(self, learning_rate: float = None, momentum: float = None):
        """
        Update graph variables setting giving `learning_rate` and `momentum`

        Args:
            learning_rate: learning rate value to be set in graph (set if not None)
            momentum: momentum value to be set in graph (set if not None)

        Returns:
            None
        """
        if learning_rate is not None:
            K.set_value(self.get_learning_rate_variable(), learning_rate)
            # log.info(f"Learning rate = {learning_rate}")
        if momentum is not None:
            K.set_value(self.get_momentum_variable(), momentum)
            # log.info(f"Momentum      = {momentum}")

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
        if (isinstance(self.opt.get("learning_rate", None), float) and
                isinstance(self.opt.get("learning_rate_decay", None), float)):
            pass
        else:
            if event_name == 'after_train_log':
                if (self.get_learning_rate_variable() is not None) and ('learning_rate' not in data):
                    data['learning_rate'] = float(K.get_value(self.get_learning_rate_variable()))
                    # data['learning_rate'] = self._lr
                if (self.get_momentum_variable() is not None) and ('momentum' not in data):
                    data['momentum'] = float(K.get_value(self.get_momentum_variable()))
                    # data['momentum'] = self._mom
            else:
                super().process_event(event_name, data)
