"""
Here is an abstract class for neural network models based on Tensorflow.
If you use something different, ex. Pytorch, then write similar to this class, inherit it from
Trainable and Inferable interfaces and make a pull-request to deeppavlov.
"""

from abc import abstractmethod

import tensorflow as tf
from overrides import overrides
from collections import defaultdict
import numpy as np

from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_attr_true, check_path_exists
from .tf_backend import TfModelMeta


class TFModel(Trainable, Inferable, metaclass=TfModelMeta):
    _saver = tf.train.Saver
    _model_dir = ''
    _model_file = ''
    sess = None

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
        if not self.train_now:
            self.load()
        return self._forward(instance, *args)

    def save(self):
        print("Saving model to `{}`".format(self.model_path_.as_posix()))
        self._saver().save(sess=self.sess, save_path=self.model_path_.as_posix(), global_step=0)
        print('\n:: Model saved to {} \n'.format(self.model_path_.as_posix()))

    def get_checkpoint_state(self):
        return tf.train.get_checkpoint_state(self.model_path_.parent)

    @check_path_exists('dir')
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
    def train_on_batch(self, batch_x, batch_y):
        """ Perform single update of trainable parameters given a batch of samples

        Args:
            batch_x: a list of input parameters or a single
                 input parameter (all are tensors ready to fed
                 to the network)
            batch_y: a lit of output parameters or a single output parameter

        Returns:
            loss: scalar loss for this batch before update of the parameters

        """
        pass

    def save(self, model_file_path):
        """
        Save model to model_file_path
        """
        print('Saving model to {}'.format(model_file_path))
        saver = tf.train.Saver()
        saver.save(self._sess, str(model_file_path))

    def load(self, model_file_path):
        """
        Load model from the model_file_path
        """
        print('Loading model from {}'.format(model_file_path))
        saver = tf.train.Saver()
        saver.restore(self._sess, str(model_file_path))

    @staticmethod
    def print_number_of_parameters():
        """
        Print number of *trainable* parameters in the network
        """
        print('Number of parameters: ')
        vars = tf.trainable_variables()
        blocks = defaultdict(int)
        for var in vars:
            # Get the top level scope name of variable
            block_name = var.name.split('/')[0]
            number_of_parameters = np.prod(var.get_shape().as_list())
            blocks[block_name] += number_of_parameters
        for block_name in blocks:
            print(block_name, blocks[block_name])
        total_num_parameters = np.sum(list(blocks.values()))
        print('Total number of parameters equal {}'.format(total_num_parameters))
