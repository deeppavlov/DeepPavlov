"""
Here is an abstract class for neural network models based on Tensorflow.
If you use something different, ex. Pytorch, then write similar to this class, inherit it from
Trainable and Inferable interfaces and make a pull-request to deeppavlov.
"""

from abc import abstractmethod
from pathlib import Path

import tensorflow as tf
from tensorflow.python.training.saver import Saver

from deeppavlov.core.common import paths
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable


class TFModel(Trainable, Inferable):
    _saver = Saver
    _model_dir_path = ''
    _model_fpath = ''
    _model_path = Path(paths.USR_PATH).joinpath(_model_dir_path, _model_fpath)
    sess = tf.Session()

    @abstractmethod
    def _add_placeholders(self):
        """
        Add all needed placeholders for a computational graph.
        """
        pass

    @abstractmethod
    def _run_sess(self):
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

    def train(self, features, *args):
        """
        Just a wrapper for a private method.
        """
        return self._train_step(features, *args)

    def infer(self, instance, *args):
        """
        Just a wrapper for a private method.
        """
        return self._forward(instance, *args)

    def save(self):
        self._saver().save(sess=self.sess, save_path=self._model_path.as_posix(), global_step=0)
        print('\n:: Model saved to {} \n'.format(self._model_path.as_posix()))

    def load(self):
        """
        Load session from checkpoint
        """
        ckpt = tf.train.get_checkpoint_state(self._model_path.parent)
        if ckpt and ckpt.model_checkpoint_path:
            print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            self._saver().restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('\n:: <ERR> checkpoint not found! \n')
