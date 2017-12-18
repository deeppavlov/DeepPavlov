"""
Here is an abstract class for neural network models based on Tensorflow.
If you use something different, ex. Pytorch, then write similar to this class, inherit it from
Trainable and Inferable interfaces and make a pull-request to deeppavlov.
"""

from abc import abstractmethod
from pathlib import Path

import tensorflow as tf

from deeppavlov.core.common import paths
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable


class TFModel(Trainable, Inferable):
    _saver = tf.train.Saver
    _model_dir_path = ''
    _model_fpath = ''
    sess = None

    @property
    def _model_path(self):
        return Path(paths.USR_PATH).joinpath(self._model_dir_path,
                                             self._model_fpath)

    @abstractmethod
    def _add_placeholders(self):
        # It seems that there is no need in such abstracti
        """
        Add all needed placeholders for a computational graph.
        """
        pass

    @abstractmethod
    def _run_sess(self):
        """
        1. Call _build_graph()
        2. Define all computations.
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
        :param args: any other inputs, including target vector, you need to
            pass for training
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
        fname = self._saver().save(sess=self.sess,
                                   save_path=self._model_path.as_posix(),
                                   global_step=0)
        print('\n:: Model saved to {} \n'.format(fname))

    def get_checkpoint_state(self, dir_name=None):
        dir_name = dir_name or self._model_path.parent
        return tf.train.get_checkpoint_state(dir_name)

    def load(self, fname=None):
        """
        Load session from fname or from checkpoint
        """
        if fname is None:
            ckpt = self.get_checkpoint_state()
            if ckpt and ckpt.model_checkpoint_path:
                fname = ckpt.model_checkpoint_path
        if fname is None:
            raise FileNotFoundError('\n:: <ERR> checkpoint not found! \n')
        print('\n:: restoring checkpoint from', fname, '\n')
        self._saver().restore(self.sess, fname)
