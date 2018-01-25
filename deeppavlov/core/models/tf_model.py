"""
Here is an abstract class for neural network models based on Tensorflow.
If you use something different, ex. Pytorch, then write similar to this class, inherit it from
Trainable and Inferable interfaces and make a pull-request to deeppavlov.
"""

from abc import abstractmethod
from collections import defaultdict
import numpy as np

import tensorflow as tf
from overrides import overrides

from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.common.attributes import check_attr_true, check_path_exists
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
        save_path = self.ser_path
        if save_path.is_dir():
            save_path = save_path / self._ser_file
        elif save_path.parent.is_dir():
            pass
        else:
            raise ConfigError("Provided ser path doesn't exists")
        print('\n:: saving model to {} \n'.format(save_path))
        self._saver().save(sess=self.sess, save_path=str(save_path), global_step=0)
        print('model saved')

    def get_checkpoint_state(self):
        if self.ser_path.is_dir():
            return tf.train.get_checkpoint_state(self.ser_path)
        else:
            return tf.train.get_checkpoint_state(self.ser_path.parent)

    @check_path_exists()
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
        ser_path = kwargs.get('ser_path', None)
        ser_dir = kwargs.get('ser_dir', 'model')
        ser_file = kwargs.get('ser_file', 'tf_model')
        train_now = kwargs.get('train_now', False)
        super().__init__(ser_path=ser_path,
                         ser_dir=ser_dir,
                         ser_file=ser_file,
                         train_now=train_now,
                         mode=kwargs['mode'])

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

    def get_train_op(self, loss, learning_rate, learnable_scopes=None, optimizer=None):
        """ Get train operation for given loss

        Args:
            loss: loss, tf tensor or scalar
            learning_rate: scalar or placeholder
            learnable_scopes: which scopes are trainable (None for all)
            optimizer: instance of tf.train.Optimizer, default Adam

        Returns:
            train_op
        """
        variables = self.get_trainable_variables(learnable_scopes)
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        # For batch norm it is necessary to update running averages
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer(learning_rate).minimize(loss, var_list=variables)
        return train_op

