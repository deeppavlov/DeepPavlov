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
from typing import Iterable, Optional, Any, Union, List, Tuple
from enum import IntEnum
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import cls_from_str
from .tf_backend import TfModelMeta


log = get_logger(__name__)


class TFModel(NNModel, metaclass=TfModelMeta):
    """Parent class for all components using TensorFlow."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self, exclude_scopes: Optional[Iterable] = ('Optimizer',)) -> None:
        """Load model parameters from self.load_path"""
        if not hasattr(self, 'sess'):
            raise RuntimeError('Your TensorFlow model {} must'
                               ' have sess attribute!'.format(self.__class__.__name__))
        path = str(self.load_path.resolve())
        # Check presence of the model files
        if tf.train.checkpoint_exists(path):
            log.info('[loading model from {}]'.format(path))
            # Exclude optimizer variables from saved variables
            var_list = self._get_saveable_variables(exclude_scopes)
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, path)

    def save(self, exclude_scopes: Optional[Iterable] = ('Optimizer',)) -> None:
        """Save model parameters to self.save_path"""
        if not hasattr(self, 'sess'):
            raise RuntimeError('Your TensorFlow model {} must'
                               ' have sess attribute!'.format(self.__class__.__name__))
        path = str(self.save_path.resolve())
        log.info('[saving model to {}]'.format(path))
        var_list = self._get_saveable_variables(exclude_scopes)
        saver = tf.train.Saver(var_list)
        saver.save(self.sess, path)

    @staticmethod
    def _get_saveable_variables(exclude_scopes=tuple()):
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
                optimizer = tf.train.AdamOptimizer(learning_rate)

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


class DecayType(IntEnum):
    ''' Data class, each decay type is assigned a number. '''
    NO = 1
    LINEAR = 2
    COSINE = 3
    EXPONENTIAL = 4
    POLYNOMIAL = 5
    ONECYCLE = 6

    @classmethod
    def from_str(cls, label: str):
        label_norm = label.replace('1', 'one').upper()
        if label_norm in cls.__members__:
            return DecayType[label_norm]
        else:
            raise NotImplementedError


class DecayScheduler():
    '''
    Given initial and endvalue, this class generates the next value
    depending on decay type and number of iterations. (by calling next_val().)
    '''

    def __init__(self, dec_type: Union[str, DecayType], start_val: float,
                 num_it: int = None, end_val: float = None, extra: float = None):
        if isinstance(dec_type, DecayType):
            self.dec_type = dec_type
        else:
            self.dec_type = DecayType.from_str(dec_type)
        self.nb, self.extra = num_it, extra
        self.start_val, self.end_val = start_val, end_val
        self.iters = 0
        if self.end_val is None and not (self.dec_type in [1, 4]):
            self.end_val = 0
        if self.dec_type == DecayType.ONECYCLE:
            self.cycle_nb = math.ceil(self.nb / 2)
            self.div = self.end_val / self.start_val

    def __str__(self):
        return f"DecayScheduler(start_val={self.start_val}, end_val={self.end_val}"\
            f", dec_type={self.dec_type.name}, num_it={self.nb}, extra={self.extra})"

    def next_val(self):
        self.iters = min(self.iters + 1, self.nb)
        # print(f"iters = {self.iters}/{self.nb}")
        if self.dec_type == DecayType.NO:
            return self.start_val
        elif self.dec_type == DecayType.LINEAR:
            pct = self.iters / self.nb
            return self.start_val + pct * (self.end_val - self.start_val)
        elif self.dec_type == DecayType.COSINE:
            cos_out = math.cos(math.pi * self.iters / self.nb) + 1
            return self.end_val + (self.start_val - self.end_val) / 2 * cos_out
        elif self.dec_type == DecayType.EXPONENTIAL:
            ratio = self.end_val / self.start_val
            return self.start_val * (ratio ** (self.iters / self.nb))
        elif self.dec_type == DecayType.POLYNOMIAL:
            delta_val = self.start_val - self.end_val
            return self.end_val + delta_val * (1 - self.iters / self.nb) ** self.extra
        elif self.dec_type == DecayType.ONECYCLE:
            if self.iters > self.cycle_nb:
                # decaying from end_val to start_val for cycle_nb steps
                pct = 1 - (self.iters - self.cycle_nb) / self.cycle_nb
                return self.start_val * (1 + pct * (self.div - 1))
            else:
                # raising from start_val to end_val for cycle_nb steps
                pct = self.iters / self.cycle_nb
                return self.start_val * (1 + pct * (self.div - 1))


class EnhancedTFModel(TFModel):
    """TFModel anhanced with optimizer, learning rate and momentum configuration"""
    def __init__(self,
                 learning_rate: Union[float, Tuple[float, float]],
                 learning_rate_decay: Union[str, DecayType, List[Any]] = DecayType.NO,
                 learning_rate_decay_epochs: int = 0,
                 learning_rate_decay_batches: int = 0,
                 learning_rate_drop_div: float = 2.0,
                 learning_rate_drop_patience: int = None,
                 momentum: Union[float, Tuple[float, float]] = None,
                 momentum_decay: Union[str, DecayType, List[Any]] = DecayType.NO,
                 momentum_decay_epochs: int = 0,
                 momentum_decay_batches: int = 0,
                 optimizer: str = 'AdamOptimizer',
                 clip_norm: float = None,
                 fit_beta: float = 0.98,
                 fit_learning_rate_div: float = 10.,
                 fit_min_batches: int = 10,
                 fit_max_batches: int = None,
                 *args, **kwargs) -> None:
        if learning_rate_decay_epochs and learning_rate_decay_batches:
            raise ConfigError("isn't able to update learning rate every batch"
                              " and every epoch sumalteniously")
        if momentum_decay_epochs and momentum_decay_batches:
            raise ConfigError("isn't able to update momentum every batch"
                              " and every epoch sumalteniously")
        super().__init__(*args, **kwargs)

        try:
            self._optimizer = cls_from_str(optimizer)
        except Exception:
            self._optimizer = getattr(tf.train, optimizer.split(':')[-1])
        if not issubclass(self._optimizer, tf.train.Optimizer):
            raise ConfigError("`optimizer` should be tensorflow.train.Optimizer subclass")

        start_val, end_val = learning_rate, None
        if isinstance(learning_rate, (tuple, list)):
            start_val, end_val = learning_rate
        dec_type, extra = learning_rate_decay, None
        if isinstance(learning_rate_decay, (tuple, list)):
            dec_type, extra = learning_rate_decay

        self._lr = start_val
        num_it, self._lr_update_on_batch = learning_rate_decay_epochs, False
        if learning_rate_decay_batches > 0:
            num_it, self._lr_update_on_batch = learning_rate_decay_batches, True

        self._lr_schedule = DecayScheduler(start_val=start_val, end_val=end_val,
                                           num_it=num_it, dec_type=dec_type, extra=extra)
        self._lr_ph = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        if (momentum is None) and\
                self._optimizer not in (tf.train.AdagradDAOptimizer,
                                        tf.train.AdagradOptimizer,
                                        tf.train.GradientDescentOptimizer,
                                        tf.train.ProximalGradientDescentOptimizer,
                                        tf.train.ProximalAdagradOptimizer):
            momentum = 0.9
        start_val, end_val = momentum, None
        if isinstance(momentum, (tuple, list)):
            start_val, end_val = momentum
        dec_type, extra = momentum_decay, None
        if isinstance(momentum_decay, (tuple, list)):
            dec_type, extra = momentum_decay

        self._mom = start_val
        num_it, self._mom_update_on_batch = momentum_decay_epochs, False
        if momentum_decay_batches > 0:
            num_it, self._mom_update_on_batch = momentum_decay_batches, True

        self._mom_schedule = DecayScheduler(start_val=start_val, end_val=end_val,
                                            num_it=num_it, dec_type=dec_type,
                                            extra=extra)
        self._mom_ph = tf.placeholder_with_default(0.9, shape=[], name='momentum')
        # self._mom_ph = tf.placeholder(tf.float32, shape=[], name='momentum')

        self._learning_rate_drop_patience = learning_rate_drop_patience
        self._learning_rate_drop_div = learning_rate_drop_div
        self._learning_rate_cur_impatience = 0.
        self._learning_rate_last_impatience = 0.
        self._learning_rate_cur_div = 1.
        self._clip_norm = clip_norm
        self._fit_beta = fit_beta
        self._fit_lr_div = fit_learning_rate_div
        self._fit_min_batches = fit_min_batches
        self._fit_max_batches = fit_max_batches

    def fit_batches(self, *args):
        self.save()
        batches = list(args)
        num_batches = self._fit_max_batches or len(batches)

        avg_loss = 0.
        best_loss = float('inf')
        lrs, losses = [], []
        self._mom = 0. if self._mom is not None else None
        _lr_find_schedule = DecayScheduler(start_val=self._lr_schedule.start_val,
                                           end_val=self._lr_schedule.end_val,
                                           dec_type="exponential",
                                           num_it=num_batches)
        self._lr = _lr_find_schedule.start_val
        best_lr = _lr_find_schedule.start_val
        break_flag = False
        i = 0
        while True:
            for batch in batches:
                i += 1
                report = self.train_on_batch(*batch)
                if not isinstance(report, dict):
                    report = {'loss': report}
                # Calculating smoothed loss
                avg_loss = self._fit_beta*avg_loss + (1 - self._fit_beta)*report['loss']
                smoothed_loss = avg_loss / (1 - self._fit_beta**i)
                lrs.append(self._lr)
                losses.append(smoothed_loss)
                log.info(f"Batch {i}/{num_batches}: smooth_loss = {smoothed_loss}"
                         f", lr = {self._lr}, best_lr = {best_lr}")
                if math.isnan(smoothed_loss) or (smoothed_loss > 4 * best_loss):
                    break_flag = True
                    break
                if (smoothed_loss < best_loss) and (i >= self._fit_min_batches):
                    best_loss = smoothed_loss
                    best_lr = self._lr
                self._lr = _lr_find_schedule.next_val()

                if i >= num_batches:
                    break_flag = True
                    break
            if break_flag:
                break
        # best_lr /= 10
        best_lr = self._get_best(lrs, losses)

        self._lr_schedule = DecayScheduler(start_val=best_lr / self._fit_lr_div,
                                           end_val=best_lr,
                                           num_it=self._lr_schedule.nb,
                                           dec_type=self._lr_schedule.dec_type,
                                           extra=self._lr_schedule.extra)
        log.info(f"Found best learning rate value = {best_lr}"
                 f", setting new learning rate schedule with {self._lr_schedule}.")
        self._lr = self._lr_schedule.start_val
        self._mom = self._mom_schedule.start_val
        self.load()
        return {'smoothed_loss': losses, 'learning_rate': lrs}

    @staticmethod
    def _get_best(values, losses, max_loss_div=0.9, min_val_div=10.0):
        assert len(values) == len(losses), "lengths of values and losses should be equal"
        min_ind = np.argmin(losses)
        for i in range(min_ind - 1, 0, -1):
            if (losses[i] * max_loss_div > losses[min_ind]) or\
                    (values[i] * min_val_div < values[min_ind]):
                return values[i + 1]
        return values[min_ind] / min_val_div

    def get_train_op(self,
                     *args,
                     learning_rate: Union[float, tf.placeholder] = None,
                     optimizer: tf.train.Optimizer = None,
                     momentum: Union[float, tf.placeholder] = None,
                     clip_norm: float = None,
                     **kwargs):
        kwargs['learning_rate'] = learning_rate or self.get_learning_rate_ph()
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
            kwargs[momentum_param] = self.get_momentum_ph()
        return super().get_train_op(*args, **kwargs)

    def process_event(self, event_name, data):
        if event_name == "after_validation":
            if data['impatience'] > self._learning_rate_last_impatience:
                self._learning_rate_cur_impatience += 1
            else:
                self._learning_rate_cur_impatience = 0

            self._learning_rate_last_impatience = data['impatience']

            if (self._learning_rate_drop_patience is not None) and\
                    (self._learning_rate_cur_impatience >=
                     self._learning_rate_drop_patience):
                self._learning_rate_cur_impatience = 0
                self._learning_rate_cur_div *= self._learning_rate_drop_div
                self._lr /= self._learning_rate_drop_div
                log.info(f"New learning rate dividor = {self._learning_rate_cur_div}")
        if event_name == 'after_batch':
            if self._lr_update_on_batch:
                self._lr = self._lr_schedule.next_val() / self._learning_rate_cur_div
            if self._mom_update_on_batch and (self.get_momentum() is not None):
                self._mom = min(1., max(0., self._mom_schedule.next_val()))
        if event_name == 'after_epoch':
            if not self._lr_update_on_batch:
                self._lr = self._lr_schedule.next_val() / self._learning_rate_cur_div
            if not self._mom_update_on_batch and (self.get_momentum() is not None):
                self._mom = min(1., max(0., self._mom_schedule.next_val()))

    def get_learning_rate(self):
        return self._lr

    def get_learning_rate_ph(self):
        return self._lr_ph

    def get_momentum(self):
        return self._mom

    def get_momentum_ph(self):
        return self._mom_ph

    def get_optimizer(self):
        return self._optimizer
