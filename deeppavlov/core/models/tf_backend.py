import tensorflow as tf
from abc import ABCMeta
from functools import wraps

from six import with_metaclass


def _graph_wrap(func, graph):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        with graph.as_default():
            return func(*args, **kwargs)
    return _wrapped


class TfModelMeta(with_metaclass(type, ABCMeta)):
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
