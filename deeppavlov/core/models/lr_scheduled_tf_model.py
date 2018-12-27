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

from typing import Iterable, Optional, Any, Union, List, Tuple
from enum import IntEnum
import math

import numpy as np
import tensorflow as tf

from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import cls_from_str
from .tf_model import TFModel


log = get_logger(__name__)


class DecayType(IntEnum):
    ''' Data class, each decay type is assigned a number. '''
    NO = 1
    LINEAR = 2
    COSINE = 3
    EXPONENTIAL = 4
    POLYNOMIAL = 5
    ONECYCLE = 6
    TRAPEZOID = 7

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
            self.div = 1.0 if not self.start_val else self.end_val / self.start_val
        if self.dec_type == DecayType.TRAPEZOID:
            self.div = 1.0 if not self.start_val else self.end_val / self.start_val

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
        elif self.dec_type == DecayType.TRAPEZOID:
            if self.iters > 0.6 * self.nb:
                # decaying from end_val to start_val for 4/10 * nb steps
                pct = 2.5 * (self.nb - self.iters) / self.nb
                return self.start_val * (1 + pct * (self.div - 1))
            elif self.iters > 0.1 * self.nb:
                # constant end_val
                return self.end_val
            else:
                # raising from start_val to end_val for 1/10 * nb steps
                pct = 10.0 * self.iters / self.nb
                return self.start_val * (1 + pct * (self.div - 1))


DType = Union[str, DecayType]


class LRScheduledTFModel(TFModel):
    """
    TFModel enhanced with optimizer, learning rate and momentum
    management and search.
    """
    def __init__(self,
                 learning_rate: Union[float, Tuple[float, float]] = None,
                 learning_rate_decay: Union[DType, Tuple[DType, Any]] = DecayType.NO,
                 learning_rate_decay_epochs: int = 0,
                 learning_rate_decay_batches: int = 0,
                 learning_rate_drop_div: float = 2.0,
                 learning_rate_drop_patience: int = None,
                 momentum: Union[float, Tuple[float, float]] = None,
                 momentum_decay: Union[DType, Tuple[DType, Any]] = DecayType.NO,
                 momentum_decay_epochs: int = 0,
                 momentum_decay_batches: int = 0,
                 optimizer: str = 'AdamOptimizer',
                 clip_norm: float = None,
                 fit_batch_size: Union[int, str] = None,
                 fit_learning_rate: Tuple[float, float] = [1e-7, 100],
                 fit_learning_rate_div: float = 10.,
                 fit_beta: float = 0.98,
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
        #self._lr_var = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self._lr_var = tf.Variable(self._lr or 0., dtype=tf.float32, name='learning_rate')

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
        # self._mom_var = tf.placeholder_with_default(0.9, shape=[], name='momentum')
        # self._mom_var = tf.placeholder(tf.float32, shape=[], name='momentum')
        self._mom_var = tf.Variable(self._mom or 0., dtype=tf.float32, name='momentum')

        self._learning_rate_drop_patience = learning_rate_drop_patience
        self._learning_rate_drop_div = learning_rate_drop_div
        self._learning_rate_cur_impatience = 0.
        self._learning_rate_last_impatience = 0.
        self._learning_rate_cur_div = 1.
        self._clip_norm = clip_norm
        self._fit_batch_size = fit_batch_size
        self._fit_learning_rate = fit_learning_rate
        self._fit_learning_rate_div = fit_learning_rate_div
        self._fit_beta = fit_beta
        self._fit_min_batches = fit_min_batches
        self._fit_max_batches = fit_max_batches
        self._external_lr = False
        self._external_mom = False

    def load(self, exclude_scopes: Optional[Iterable] = ('Optimizer',
                                                         'learning_rate',
                                                         'momentum')):
        return super().load(exclude_scopes=exclude_scopes)

    def fit(self, *args):
        data = list(zip(*args))
        self.save()
        if self._fit_batch_size is None:
            raise ConfigError("in order to use fit() method"
                              " set `fit_batch_size` parameter")
        bs = int(self._fit_batch_size)
        data_len = len(data)
        num_batches = self._fit_max_batches or ((data_len - 1) // bs + 1)

        avg_loss = 0.
        best_loss = float('inf')
        lrs, losses = [], []
        _lr_find_schedule = DecayScheduler(start_val=self._fit_learning_rate[0],
                                           end_val=self._fit_learning_rate[1],
                                           dec_type="exponential",
                                           num_it=num_batches)
        self._lr = _lr_find_schedule.start_val
        self._mom = 0.
        self._update_tf_variables(learning_rate=self._lr, momentum=self._mom)
        best_lr = _lr_find_schedule.start_val
        for i in range(num_batches):
            batch_start = (i * bs) % data_len
            batch_end = batch_start + bs
            report = self.train_on_batch(*zip(*data[batch_start:batch_end]))
            if not isinstance(report, dict):
                report = {'loss': report}
            # Calculating smoothed loss
            avg_loss = self._fit_beta*avg_loss + (1 - self._fit_beta)*report['loss']
            smoothed_loss = avg_loss / (1 - self._fit_beta**(i + 1))
            lrs.append(self._lr)
            losses.append(smoothed_loss)
            log.info(f"Batch {i}/{num_batches}: smooth_loss = {smoothed_loss}"
                     f", lr = {self._lr}, best_lr = {best_lr}")
            if math.isnan(smoothed_loss) or (smoothed_loss > 4 * best_loss):
                break
            if (smoothed_loss < best_loss) and (i >= self._fit_min_batches):
                best_loss = smoothed_loss
                best_lr = self._lr
            self._lr = _lr_find_schedule.next_val()
            self._update_tf_variables(learning_rate=self._lr)

            if i >= num_batches:
                break
        # best_lr /= 10
        end_val = self._get_best(lrs, losses)

        start_val = end_val
        if self._lr_schedule.dec_type in (DecayType.ONECYCLE, DecayType.TRAPEZOID):
            start_val = end_val / self._fit_learning_rate_div
        elif self._lr_schedule.dec_type in (DecayType.POLYNOMIAL, DecayType.EXPONENTIAL,
                                            DecayType.LINEAR):
            start_val = end_val
            end_val = end_val / self._fit_learning_rate_div
        self._lr_schedule = DecayScheduler(start_val=start_val,
                                           end_val=end_val,
                                           num_it=self._lr_schedule.nb,
                                           dec_type=self._lr_schedule.dec_type,
                                           extra=self._lr_schedule.extra)
        log.info(f"Found best learning rate value = {best_lr}"
                 f", setting new learning rate schedule with {self._lr_schedule}.")

        self.load()
        self._lr = self._lr_schedule.start_val
        self._mom = self._mom_schedule.start_val
        self._update_tf_variables(learning_rate=self._lr, momentum=self._mom)
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
        if learning_rate is not None:
            self._external_lr = True
            kwargs['learning_rate'] = learning_rate
        else:
            kwargs['learning_rate'] = self.get_learning_rate_var()
        kwargs['optimizer'] = optimizer or self.get_optimizer()
        kwargs['clip_norm'] = clip_norm or self._clip_norm

        momentum_param = 'momentum'
        if kwargs['optimizer'] == tf.train.AdamOptimizer:
            momentum_param = 'beta1'
        elif kwargs['optimizer'] == tf.train.AdadeltaOptimizer:
            momentum_param = 'rho'

        if momentum is not None:
            self._external_mom = True
            kwargs[momentum_param] = momentum
        elif self.get_momentum() is not None:
            kwargs[momentum_param] = self.get_momentum_var()
        return super().get_train_op(*args, **kwargs)

    def _update_tf_variables(self, learning_rate=None, momentum=None):
        if learning_rate is not None:
            self.sess.run(tf.assign(self._lr_var, learning_rate))
            #log.info(f"Learning rate = {learning_rate}")
        if momentum is not None:
            self.sess.run(tf.assign(self._mom_var, momentum))
            #log.info(f"Momentum      = {momentum}")

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
                self._update_tf_variables(learning_rate=self._lr)
                log.info(f"New learning rate dividor = {self._learning_rate_cur_div}")
        if event_name == 'after_batch':
            if (self._lr is not None) and self._lr_update_on_batch:
                self._lr = self._lr_schedule.next_val() / self._learning_rate_cur_div
                self._update_tf_variables(learning_rate=self._lr)
            if (self._mom is not None) and self._mom_update_on_batch:
                self._mom = min(1., max(0., self._mom_schedule.next_val()))
                self._update_tf_variables(momentum=self._mom)
        if event_name == 'after_epoch':
            if (self._lr is not None) and not self._lr_update_on_batch:
                self._lr = self._lr_schedule.next_val() / self._learning_rate_cur_div
                self._update_tf_variables(learning_rate=self._lr)
            if (self._mom is not None) and not self._mom_update_on_batch:
                self._mom = min(1., max(0., self._mom_schedule.next_val()))
                self._update_tf_variables(momentum=self._mom)
        if event_name == 'after_train_log':
            if (self._lr is not None) and not self._external_lr:
                data['learning_rate'] = self._lr
            if (self._mom is not None) and not self._external_mom:
                data['momentum'] = self._mom

    def get_learning_rate(self):
        if self._lr is None:
            raise ConfigError("Please specify `learning_rate` parameter"
                              " before training")
        return self._lr

    def get_learning_rate_var(self):
        return self._lr_var

    def get_momentum(self):
        return self._mom

    def get_momentum_var(self):
        return self._mom_var

    def get_optimizer(self):
        return self._optimizer
