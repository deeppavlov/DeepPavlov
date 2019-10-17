# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Iterable, Union, Tuple, Optional

import numpy as np
import tensorflow as tf
from overrides import overrides
from tensorflow.python.ops import variables

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import cls_from_str
from deeppavlov.core.models.lr_scheduled_model import LRScheduledModel
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.models.tf_backend import TfModelMeta

log = getLogger(__name__)


class TFModel(NNModel, metaclass=TfModelMeta):
    """Parent class for all components using TensorFlow."""

    sess: tf.Session

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self, exclude_scopes: tuple = ('Optimizer',), path: Union[Path, str] = None) -> None:
        """Load model parameters from self.load_path"""
        if not hasattr(self, 'sess'):
            raise RuntimeError('Your TensorFlow model {} must'
                               ' have sess attribute!'.format(self.__class__.__name__))
        path = path or self.load_path
        path = str(Path(path).resolve())
        # Check presence of the model files
        if tf.train.checkpoint_exists(path):
            log.info('[loading model from {}]'.format(path))
            # Exclude optimizer variables from saved variables
            var_list = self._get_saveable_variables(exclude_scopes)
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, path)

    def deserialize(self, weights: Iterable[Tuple[str, np.ndarray]]) -> None:
        assign_ops = []
        feed_dict = {}
        for var_name, value in weights:
            var = self.sess.graph.get_tensor_by_name(var_name)
            value = np.asarray(value)
            assign_placeholder = tf.placeholder(var.dtype, shape=value.shape)
            assign_op = tf.assign(var, assign_placeholder)
            assign_ops.append(assign_op)
            feed_dict[assign_placeholder] = value
        self.sess.run(assign_ops, feed_dict=feed_dict)

    def save(self, exclude_scopes: tuple = ('Optimizer',)) -> None:
        """Save model parameters to self.save_path"""
        if not hasattr(self, 'sess'):
            raise RuntimeError('Your TensorFlow model {} must'
                               ' have sess attribute!'.format(self.__class__.__name__))
        path = str(self.save_path.resolve())
        log.info('[saving model to {}]'.format(path))
        var_list = self._get_saveable_variables(exclude_scopes)
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, path)

    def serialize(self) -> Tuple[Tuple[str, np.ndarray], ...]:
        tf_vars = tf.global_variables()
        values = self.sess.run(tf_vars)
        return tuple(zip([var.name for var in tf_vars], values))

    @staticmethod
    def _get_saveable_variables(exclude_scopes=tuple()):
        # noinspection PyProtectedMember
        all_vars = variables._all_saveable_objects()
        vars_to_train = [var for var in all_vars if all(sc not in var.name for sc in exclude_scopes)]
        return vars_to_train

    @staticmethod
    def _get_trainable_variables(exclude_scopes=tuple()):
        all_vars = tf.global_variables()
        vars_to_train = [var for var in all_vars if all(sc not in var.name for sc in exclude_scopes)]
        return vars_to_train

    def get_train_op(self,
                     loss,
                     learning_rate,
                     optimizer=None,
                     clip_norm=None,
                     learnable_scopes=None,
                     optimizer_scope_name=None,
                     **kwargs):
        """
        Get train operation for given loss

        Args:
            loss: loss, tf tensor or scalar
            learning_rate: scalar or placeholder.
            clip_norm: clip gradients norm by clip_norm.
            learnable_scopes: which scopes are trainable (None for all).
            optimizer: instance of tf.train.Optimizer, default Adam.
            **kwargs: parameters passed to tf.train.Optimizer object
               (scalars or placeholders).

        Returns:
            train_op
        """
        if optimizer_scope_name is None:
            opt_scope = tf.variable_scope('Optimizer')
        else:
            opt_scope = tf.variable_scope(optimizer_scope_name)
        with opt_scope:
            if learnable_scopes is None:
                variables_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                variables_to_train = []
                for scope_name in learnable_scopes:
                    variables_to_train.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name))

            if optimizer is None:
                optimizer = tf.train.AdamOptimizer

            # For batch norm it is necessary to update running averages
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):

                def clip_if_not_none(grad):
                    if grad is not None:
                        return tf.clip_by_norm(grad, clip_norm)

                opt = optimizer(learning_rate, **kwargs)
                grads_and_vars = opt.compute_gradients(loss, var_list=variables_to_train)
                if clip_norm is not None:
                    grads_and_vars = [(clip_if_not_none(grad), var)
                                      for grad, var in grads_and_vars]
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
        for block_name, cnt in blocks.items():
            log.info("{} - {}.".format(block_name, cnt))
        total_num_parameters = np.sum(list(blocks.values()))
        log.info('Total number of parameters equal {}'.format(total_num_parameters))

    def destroy(self):
        if hasattr(self, 'sess'):
            for k in list(self.sess.graph.get_all_collection_keys()):
                self.sess.graph.clear_collection(k)
        super().destroy()


class LRScheduledTFModel(TFModel, LRScheduledModel):
    """
    TFModel enhanced with optimizer, learning rate and momentum
    management and search.
    """

    def __init__(self,
                 optimizer: str = 'AdamOptimizer',
                 clip_norm: float = None,
                 momentum: float = None,
                 **kwargs) -> None:
        TFModel.__init__(self, **kwargs)

        try:
            self._optimizer = cls_from_str(optimizer)
        except Exception:
            self._optimizer = getattr(tf.train, optimizer.split(':')[-1])
        if not issubclass(self._optimizer, tf.train.Optimizer):
            raise ConfigError("`optimizer` should be tensorflow.train.Optimizer subclass")
        self._clip_norm = clip_norm

        LRScheduledModel.__init__(self, momentum=momentum, **kwargs)

    @overrides
    def _init_learning_rate_variable(self):
        return tf.Variable(self._lr or 0., dtype=tf.float32, name='learning_rate')

    @overrides
    def _init_momentum_variable(self):
        return tf.Variable(self._mom or 0., dtype=tf.float32, name='momentum')

    @overrides
    def _update_graph_variables(self, learning_rate=None, momentum=None):
        if learning_rate is not None:
            self.sess.run(tf.assign(self._lr_var, learning_rate))
            # log.info(f"Learning rate = {learning_rate}")
        if momentum is not None:
            self.sess.run(tf.assign(self._mom_var, momentum))
            # log.info(f"Momentum      = {momentum}")

    def get_train_op(self,
                     loss,
                     learning_rate: Union[float, tf.placeholder] = None,
                     optimizer: tf.train.Optimizer = None,
                     momentum: Union[float, tf.placeholder] = None,
                     clip_norm: float = None,
                     **kwargs):
        if learning_rate is not None:
            kwargs['learning_rate'] = learning_rate
        else:
            kwargs['learning_rate'] = self._lr_var
        kwargs['optimizer'] = optimizer or self.get_optimizer()
        kwargs['clip_norm'] = clip_norm or self._clip_norm

        momentum_param = 'momentum'
        if kwargs['optimizer'] == tf.train.AdamOptimizer:
            momentum_param = 'beta1'
        elif kwargs['optimizer'] == tf.train.AdadeltaOptimizer:
            momentum_param = 'rho'

        if momentum is not None:
            kwargs[momentum_param] = momentum
        elif self.get_momentum() is not None:
            kwargs[momentum_param] = self._mom_var
        return TFModel.get_train_op(self, loss, **kwargs)

    def get_optimizer(self):
        return self._optimizer

    def load(self,
             exclude_scopes: Optional[Iterable] = ('Optimizer',
                                                   'learning_rate',
                                                   'momentum'),
             **kwargs):
        return super().load(exclude_scopes=exclude_scopes, **kwargs)

    def process_event(self, *args, **kwargs):
        LRScheduledModel.process_event(self, *args, **kwargs)
