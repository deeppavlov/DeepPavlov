"""
Here is an abstract class for neural network models based on Tensorflow.
If you use something different, ex. Pytorch, then write similar to this class, inherit it from
Trainable and Inferable interfaces and make a pull-request to deeppavlov.
"""

from abc import abstractmethod
from warnings import warn

import tensorflow as tf
from overrides import overrides

from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.errors import ConfigError
from .tf_backend import TfModelMeta


class TFModel(Trainable, Inferable, metaclass=TfModelMeta):
    def __init__(self, **kwargs):
        self._saver = tf.train.Saver
        super().__init__(**kwargs)

    @abstractmethod
    def _add_placeholders(self):
        """
        Add all needed placeholders for a computational graph.
        """
        pass

    @abstractmethod
    def run_sess(self, *args, **kwargs):
        """
        1. Call _build_graph()
        2. Define all comuptations.
        3. Run tf.sess.
        3. Reset state if needed.
        :return:
        """
        pass

    @abstractmethod
    def _train_step(self, features, *args):
        """
        Define a single training step. Feed dict to tf session.
        :param features: input features
        :param args: any other inputs, including target vector, you need to pass for training
        :return: metric to return, usually loss
        """
        pass

    @abstractmethod
    def _forward(self, features, *args):
        """
        Pass an instance to get a prediction.
        :param features: input features
        :param args: any other inputs you need to pass for training
        :return: prediction
        """
        pass

    @check_attr_true('train_now')
    def train(self, features, *args, **kwargs):
        """
        Just a wrapper for a private method.
        """
        return self._train_step(features, *args, **kwargs)

    def infer(self, instance, *args):
        """
        Just a wrapper for a private method.
        """
        return self._forward(instance, *args)

    def save(self):
        save_path = str(self.save_path)
        saver = tf.train.Saver()
        print('Saving NerNetwork model to {}'.format(save_path))
        saver.save(self.sess, save_path)

    def get_checkpoint_state(self):
        if self.load_path:
            if self.load_path.parent.is_dir():
                return tf.train.get_checkpoint_state(self.load_path.parent)
            else:
                warn('Provided `load_path` is incorrect!')
        else:
            warn("No `load_path` is provided for {}".format(self.__class__.__name__))

    @overrides
    def load(self):
        """
        Load session from checkpoint
        """
        ckpt = self.get_checkpoint_state()
        if ckpt and ckpt.model_checkpoint_path:
            print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            self._saver().restore(self.sess, ckpt.model_checkpoint_path)
            print('session restored')
        else:
            print('\n:: <ERR> checkpoint not found! \n')


class SimpleTFModel(Trainable, Inferable, metaclass=TfModelMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
