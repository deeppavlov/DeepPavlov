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
from collections import defaultdict
import numpy as np

import tensorflow as tf

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.log import get_logger
from .tf_backend import TfModelMeta

log = get_logger(__name__)


class TFModel(NNModel, metaclass=TfModelMeta):
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'sess'):
            raise RuntimeError('Your tensorflow model {} must'
                               ' have sess attribute!'.format(self.__class__.__name__))
        super().__init__(*args, **kwargs)

    def load(self, exclude_scopes=['Optimizer']):
        """Load model parameters from self.load_path"""
        path = str(self.load_path.resolve())
        # Check presence of the model files
        if tf.train.checkpoint_exists(path):
            log.info('[loading model from {}]'.format(path))
            # Exclude optimizer variables from saved variables
            var_list = self._get_trainable_variables(exclude_scopes)
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, path)

    def save(self, exclude_scopes=['Optimizer']):
        """Save model parameters to self.save_path"""
        path = str(self.save_path.resolve())
        log.info('[saving model to {}]'.format(path))
        var_list = self._get_trainable_variables(exclude_scopes)
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, path)

    def _get_trainable_variables(self, exclude_scopes=[]):
        all_vars = tf.global_variables()
        vars_to_train = [var for var in all_vars if all(sc not in var.name for sc in exclude_scopes)]
        return vars_to_train

    def get_train_op(self,
                     loss,
                     learning_rate,
                     optimizer=None,
                     clip_norm=None,
                     learnable_scopes=None,
                     optimizer_scope_name=None):
        """ Get train operation for given loss

        Args:
            loss: loss, tf tensor or scalar
            learning_rate: scalar or placeholder
            clip_norm: clip gradients norm by clip_norm
            learnable_scopes: which scopes are trainable (None for all)
            optimizer: instance of tf.train.Optimizer, default Adam

        Returns:
            train_op
        """
        if optimizer_scope_name is None:
            opt_scope = tf.variable_scope('Optimizer')
        else:
            opt_scope = tf.variable_scope(optimizer_scope_name)
        with opt_scope:
            if learnable_scopes is None:
                variables_to_train = tf.global_variables()
            else:
                variables_to_train = []
                for scope_name in learnable_scopes:
                    for var in tf.global_variables():
                        if scope_name in var.name:
                            variables_to_train.append(var)

            if optimizer is None:
                optimizer = tf.train.AdamOptimizer

            # For batch norm it is necessary to update running averages
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                opt = optimizer(learning_rate)
                grads_and_vars = opt.compute_gradients(loss, var_list=variables_to_train)
                if clip_norm is not None:
                    grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var)
                                      for grad, var in grads_and_vars] #  if grad is not None
                train_op = opt.apply_gradients(grads_and_vars)
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
