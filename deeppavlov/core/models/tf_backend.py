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

from abc import ABCMeta
from functools import wraps

from six import with_metaclass
import tensorflow as tf


def _graph_wrap(func, graph):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        with graph.as_default():
            return func(*args, **kwargs)
    return _wrapped


class TfModelMeta(with_metaclass(type, ABCMeta)):
    """Metaclass that helps all child classes to have their own graph."""
    def __call__(cls, *args, **kwargs):
        from .keras_model import KerasModel
        if issubclass(cls, KerasModel):
            import keras.backend as K
            K.clear_session()

        obj = cls.__new__(cls)
        obj.graph = tf.Graph()
        for meth in dir(obj):
            if meth == '__class__':
                continue
            attr = getattr(obj, meth)
            if callable(attr):
                setattr(obj, meth, _graph_wrap(attr, obj.graph))
        obj.__init__(*args, **kwargs)
        return obj
