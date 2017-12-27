"""
Here is an abstract class for neural network models based on Tensorflow.
If you use something different, ex. Pytorch, then write similar to this class, inherit it from
Trainable and Inferable interfaces and make a pull-request to deeppavlov.
"""
import tensorflow as tf
from six import with_metaclass
from abc import ABCMeta


def _graph_wrap(func, graph):
    def _wrapped(*args, **kwargs):
        with graph.as_default():
            try:
                return func(*args, **kwargs)
            except TypeError:
                print("wrapped function is {}".format(func))
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
