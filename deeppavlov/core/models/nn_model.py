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

from typing import Tuple

from .component import Component
from .serializable import Serializable


class NNModel(Component, Serializable):
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
    def train_on_batch(self, batch: Tuple[list, list]):
        pass
