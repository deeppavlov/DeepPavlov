"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
from abc import abstractmethod
from collections import defaultdict
import numpy as np

import tensorflow as tf

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.log import get_logger
from .tf_backend import TfModelMeta

log = get_logger(__name__)


class TFModel(NNModel, metaclass=TfModelMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def train_on_batch(self, x_batch, y_batch):
        """ Perform single step of optimization given batch of samples

        Args:
            x_batch: batch of x-s it could be anything: dict, list, numpy array. However,
                     it must contain  number of samples
            y_batch: batch of y-s. It must contain same number of samples as x_batch.

        Returns:
            loss: mean loss over batch

        """
        pass

    def load(self):
        """Load model parameters from self.load_path"""
        path = str(self.load_path.resolve())
        # Check presence of the model files
        if tf.train.checkpoint_exists(path):
            print('[loading model from {}]'.format(path), file=sys.stderr)
            saver = tf.train.Saver()
            saver.restore(self.sess, path)

    def save(self):
        """Save model parameters to self.save_path"""
        path = str(self.save_path.resolve())
        print('[saving model to {}]'.format(path), file=sys.stderr)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    @abstractmethod
    def __call__(self, x_batch):
        """ Infer y_batch from x_batch

        Args:
            x_batch: a batch of samples

        Returns:
            y_batch: a batch of samples inferred from x_batch
        """
        pass

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

        if learnable_scopes is None:
            variables_to_train = tf.trainable_variables()
        else:
            variables_to_train = []
            for scope_name in learnable_scopes:
                for var in tf.trainable_variables():
                    if var.name.startswith(scope_name):
                        variables_to_train.append(var)

        if optimizer is None:
            optimizer = tf.train.AdamOptimizer

        # For batch norm it is necessary to update running averages
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer(learning_rate).minimize(loss, var_list=variables_to_train)
        return train_op

    @staticmethod
    def print_number_of_parameters():
        """
        Print number of *trainable* parameters in the network
        """
        log.info('Number of parameters: ')
        variables = tf.trainable_variables()
        blocks = defaultdict(int)
        for var in variables:
            # Get the top level scope name of variable
            block_name = var.name.split('/')[0]
            number_of_parameters = np.prod(var.get_shape().as_list())
            blocks[block_name] += number_of_parameters
        for block_name in blocks:
            log.info(block_name, blocks[block_name])
        total_num_parameters = np.sum(list(blocks.values()))
        log.info('Total number of parameters equal {}'.format(total_num_parameters))
