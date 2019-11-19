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

from abc import ABCMeta, abstractmethod

from logging import getLogger

log = getLogger(__name__)


class Component(metaclass=ABCMeta):
    """Abstract class for all callables that could be used in Chainer's pipe."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def destroy(self):
        attr_list = list(self.__dict__.keys())
        for attr_name in attr_list:
            attr = getattr(self, attr_name)
            if hasattr(attr, 'destroy'):
                attr.destroy()
            delattr(self, attr_name)

    def serialize(self):
        from deeppavlov.core.models.serializable import Serializable
        if isinstance(self, Serializable):
            log.warning(f'Method for {self.__class__.__name__} serialization is not implemented!'
                        f' Will not be able to load without using load_path')
        return None

    def deserialize(self, data):
        from deeppavlov.core.models.serializable import Serializable
        if isinstance(self, Serializable):
            log.warning(f'Method for {self.__class__.__name__} deserialization is not implemented!'
                        f' Please, use traditional load_path for this component')
        pass
