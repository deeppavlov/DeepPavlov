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

"""
:class:`deeppavlov.models.model.Trainable` is an abstract base class that expresses the interface
for all models that can be trained (ex. neural networks, scikit-learn estimators, gensim models,
etc.). All trainable models should inherit from this class.
"""

from abc import abstractmethod

from .serializable import Serializable


class Trainable(Serializable):
    """
    :attr:`train_now` expresses a developer intent for whether a model as part of a pipeline
    should be trained in the current experiment run or not.
    """

    def __init__(self, train_now=False, **kwargs):
        mode = kwargs.get('mode', None)
        if mode == 'train':
            self.train_now = train_now
        else:
            self.train_now = False
        super().__init__(**kwargs)

    @abstractmethod
    def train(self, data, *args, **kwargs):
        """
        Train a model.
        :param data: any type of input data passed for training
        :param args: all needed params for training
        As a result of training, the model should be saved to user dir defined at
        deeppavlov.common.paths.USR_PATH. A particular path is assigned in runtime.
        """
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass
