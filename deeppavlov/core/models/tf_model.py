"""
Here is an abstract class for neural network models based on Tensorflow.
If you use something different, ex. Pytorch, then write similar to this class, inherit it from
Trainable and Inferable interfaces and make a pull-request to deeppavlov.
"""

from abc import abstractmethod
from pathlib import Path

import tensorflow as tf
from tensorflow.python.training.saver import Saver
from overrides import overrides

from deeppavlov.core.common import paths
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_attr_true, run_alt_meth_if_no_path, \
    check_path_exists


class TFModel(Trainable, Inferable):
    _saver = Saver
    sess = None

    @abstractmethod
    def _add_placeholders(self):
        # It seems that there is no need in such abstracti
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
    def _build_graph(self):
        """
        Reset the default graph and add placeholders here
        Ex.:
            tf.reset_default_graph()
            self._add_placeholders()
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
        if not self.train_now:
            self.load()
        return self._forward(instance, *args)

    def save(self):
        self._saver().save(sess=self.sess, save_path=self.model_path.as_posix(), global_step=0)
        print('\n:: Model saved to {} \n'.format(self.model_path.as_posix()))

    @check_path_exists()
    def load(self):
        """
        Load session from checkpoint
        """
        ckpt = tf.train.get_checkpoint_state(self.model_path.parent)
        if ckpt and ckpt.model_checkpoint_path:
            print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            self._saver().restore(self.sess, ckpt.model_checkpoint_path)
            print('session restored')
        else:
            print('\n:: <ERR> checkpoint not found! \n')
