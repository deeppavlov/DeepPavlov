# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

import datetime
import json
import time
from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union, Optional, Iterable

from tqdm import tqdm

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log_events import get_tb_writer
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.trainers.fit_trainer import FitTrainer
from deeppavlov.core.trainers.utils import parse_metrics, NumpyArrayEncoder

log = getLogger(__name__)
report_log = getLogger('train_report')


@register('nn_trainer')
class NNTrainer(FitTrainer):
    """
    | Bases :class:`~deeppavlov.core.trainers.FitTrainer`
    | Trainer class for training and evaluating pipelines containing
      :class:`Estimators <deeppavlov.core.models.estimator.Estimator>`
      and an :class:`~deeppavlov.core.models.nn_model.NNModel`

    Args:
        chainer_config: ``"chainer"`` block of a configuration file
        batch_size: batch_size to use for partial fitting (if available) and evaluation,
            the whole dataset is used if ``batch_size`` is negative or zero (default is ``1``)
        epochs: maximum epochs number to train the pipeline, ignored if negative or zero (default is ``-1``)
        start_epoch_num: starting epoch number for reports (default is ``0``)
        max_batches: maximum batches number to train the pipeline, ignored if negative or zero (default is ``-1``)
        metrics: iterable of metrics where each metric can be a registered metric name or a dict of ``name`` and
            ``inputs`` where ``name`` is a registered metric name and ``inputs`` is a collection of parameter names
            from chainer’s inner memory that will be passed to the metric function;
            default value for ``inputs`` parameter is a concatenation of chainer’s ``in_y`` and ``out`` fields;
            the first metric is used for early stopping (default is ``('accuracy',)``)
        train_metrics: metrics calculated for train logs (if omitted, ``metrics`` argument is used)
        metric_optimization: one of ``'maximize'`` or ``'minimize'`` — strategy for metric optimization used in early
            stopping (default is ``'maximize'``)
        evaluation_targets: data types on which to evaluate a trained pipeline (default is ``('valid', 'test')``)
        show_examples: a flag used to print inputs, expected outputs and predicted outputs for the last batch
            in evaluation logs (default is ``False``)
        tensorboard_log_dir: path to a directory where tensorboard logs can be stored, ignored if None
            (default is ``None``)
        validate_first: flag used to calculate metrics on the ``'valid'`` data type before starting training
            (default is ``True``)
        validation_patience: how many times in a row the validation metric has to not improve for early stopping,
            ignored if negative or zero (default is ``5``)
        val_every_n_epochs: how often (in epochs) to validate the pipeline, ignored if negative or zero
            (default is ``-1``)
        val_every_n_batches: how often (in batches) to validate the pipeline, ignored if negative or zero
            (default is ``-1``)
        log_every_n_epochs: how often (in epochs) to calculate metrics on train data, ignored if negative or zero
            (default is ``-1``)
        log_every_n_batches: how often (in batches) to calculate metrics on train data, ignored if negative or zero
            (default is ``-1``)
        log_on_k_batches: count of random train batches to calculate metrics in log (default is ``1``)
        max_test_batches: maximum batches count for pipeline testing and evaluation, overrides ``log_on_k_batches``,
            ignored if negative (default is ``-1``)
        **kwargs: additional parameters whose names will be logged but otherwise ignored


    Trainer saves the model if it sees progress in scores. The full rules look like following:

    - For the validation savepoint:
        * 0-th validation (optional). Don't save model, establish a baseline.
        * 1-th validation.
             + If we have a baseline, save the model if we see an improvement, don't save otherwise.
             + If we don't have a baseline, save the model.
        * 2nd and later validations. Save the model if we see an improvement
    - For the at-train-exit savepoint:
        * Save the model if it happened before 1st validation (to capture early training results), don't save otherwise.

    """

    def __init__(self, chainer_config: dict, *, 
                 batch_size: int = 1,
                 epochs: int = -1,
                 start_epoch_num: int = 0,
                 max_batches: int = -1,
                 metrics: Iterable[Union[str, dict]] = ('accuracy',),
                 train_metrics: Optional[Iterable[Union[str, dict]]] = None,
                 metric_optimization: str = 'maximize',
                 evaluation_targets: Iterable[str] = ('valid', 'test'),
                 show_examples: bool = False,
                 tensorboard_log_dir: Optional[Union[str, Path]] = None,
                 max_test_batches: int = -1,
                 validate_first: bool = True,
                 validation_patience: int = 5, val_every_n_epochs: int = -1, val_every_n_batches: int = -1,
                 log_every_n_batches: int = -1, log_every_n_epochs: int = -1, log_on_k_batches: int = 1,
                 **kwargs) -> None:
        super().__init__(chainer_config, batch_size=batch_size, metrics=metrics, evaluation_targets=evaluation_targets,
                         show_examples=show_examples, max_test_batches=max_test_batches, **kwargs)
        if train_metrics is None:
            self.train_metrics = self.metrics
        else:
            self.train_metrics = parse_metrics(train_metrics, self._chainer.in_y, self._chainer.out_params)

        metric_optimization = metric_optimization.strip().lower()
        self.score_best = None

        def _improved(op):
            return lambda score, baseline: False if baseline is None or score is None \
                else op(score, baseline)

        if metric_optimization == 'maximize':
            self.improved = _improved(lambda a, b: a > b)
        elif metric_optimization == 'minimize':
            self.improved = _improved(lambda a, b: a < b)
        else:
            raise ConfigError('metric_optimization has to be one of {}'.format(['maximize', 'minimize']))

        self.validate_first = validate_first
        self.validation_number = 0 if validate_first else 1
        self.validation_patience = validation_patience
        self.val_every_n_epochs = val_every_n_epochs
        self.val_every_n_batches = val_every_n_batches
        self.log_every_n_epochs = log_every_n_epochs
        self.log_every_n_batches = log_every_n_batches
        self.log_on_k_batches = log_on_k_batches if log_on_k_batches >= 0 else None

        self.max_epochs = epochs
        self.epoch = start_epoch_num
        self.max_batches = max_batches

        self.train_batches_seen = 0
        self.examples = 0
        self.patience = 0
        self.last_result = {}
        self.losses = []
        self.start_time: Optional[float] = None
        self.tb_writer = get_tb_writer(tensorboard_log_dir)

    def save(self) -> None:
        if self._loaded:
            raise RuntimeError('Cannot save already finalized chainer')

        self._chainer.save()

    def _is_initial_validation(self):
        return self.validation_number == 0

    def _is_first_validation(self):
        return self.validation_number == 1

    def _validate(self, iterator: DataLearningIterator,
                  tensorboard_tag: Optional[str] = None, tensorboard_index: Optional[int] = None) -> None:
        self._send_event(event_name='before_validation')
        report = self.test(iterator.gen_batches(self.batch_size, data_type='valid', shuffle=False),
                           start_time=self.start_time)

        report['epochs_done'] = self.epoch
        report['batches_seen'] = self.train_batches_seen
        report['train_examples_seen'] = self.examples

        metrics = list(report['metrics'].items())

        if tensorboard_tag is not None and self.tb_writer is not None:
            if tensorboard_index is None:
                tensorboard_index = self.train_batches_seen
            for name, score in metrics:
                self.tb_writer.write_valid(tag=f'{tensorboard_tag}/{name}', scalar_value=score,
                                           global_step=tensorboard_index)
            self.tb_writer.flush()

        m_name, score = metrics[0]

        # Update the patience
        if self.score_best is None:
            self.patience = 0
        else:
            if self.improved(score, self.score_best):
                self.patience = 0
            else:
                self.patience += 1

        # Run the validation model-saving logic
        if self._is_initial_validation():
            log.info('Initial best {} of {}'.format(m_name, score))
            self.score_best = score
        elif self._is_first_validation() and self.score_best is None:
            log.info('First best {} of {}'.format(m_name, score))
            self.score_best = score
            log.info('Saving model')
            self.save()
        elif self.improved(score, self.score_best):
            log.info(f'Improved best {m_name} from {self.score_best} to {score}')
            self.score_best = score
            log.info('Saving model')
            self.save()
        else:
            log.info('Did not improve on the {} of {}'.format(m_name, self.score_best))

        report['impatience'] = self.patience
        if self.validation_patience > 0:
            report['patience_limit'] = self.validation_patience

        self._send_event(event_name='after_validation', data=report)
        report = {'valid': report}
        report_log.info(json.dumps(report, ensure_ascii=False, cls=NumpyArrayEncoder))
        self.validation_number += 1

    def _log(self, iterator: DataLearningIterator,
             tensorboard_tag: Optional[str] = None, tensorboard_index: Optional[int] = None) -> None:
        self._send_event(event_name='before_log')
        if self.log_on_k_batches == 0:
            report = {
                'time_spent': str(datetime.timedelta(seconds=round(time.time() - self.start_time + 0.5)))
            }
        else:
            data = islice(iterator.gen_batches(self.batch_size, data_type='train', shuffle=True),
                          self.log_on_k_batches)
            report = self.test(data, self.train_metrics, start_time=self.start_time)

        report.update({
            'epochs_done': self.epoch,
            'batches_seen': self.train_batches_seen,
            'train_examples_seen': self.examples
        })

        metrics: List[Tuple[str, float]] = list(report.get('metrics', {}).items()) + list(self.last_result.items())

        report.update(self.last_result)
        if self.losses:
            report['loss'] = sum(self.losses) / len(self.losses)
            self.losses.clear()
            metrics.append(('loss', report['loss']))

        if metrics and self.tb_writer is not None:
            for name, score in metrics:
                self.tb_writer.write_train(tag=f'{tensorboard_tag}/{name}', scalar_value=score,
                                           global_step=tensorboard_index)
            self.tb_writer.flush()

        self._send_event(event_name='after_train_log', data=report)

        report = {'train': report}
        report_log.info(json.dumps(report, ensure_ascii=False, cls=NumpyArrayEncoder))

    def _send_event(self, event_name: str, data: Optional[dict] = None) -> None:
        report = {
            'time_spent': str(datetime.timedelta(seconds=round(time.time() - self.start_time + 0.5))),
            'epochs_done': self.epoch,
            'batches_seen': self.train_batches_seen,
            'train_examples_seen': self.examples
        }
        if data is not None:
            report.update(data)
        self._chainer.process_event(event_name=event_name, data=report)

    def train_on_batches(self, iterator: DataLearningIterator) -> None:
        """Train pipeline on batches using provided data iterator and initialization parameters"""
        self.start_time = time.time()
        if self.validate_first:
            self._validate(iterator)

        while True:
            impatient = False
            self._send_event(event_name='before_train')
            for x, y_true in tqdm(iterator.gen_batches(self.batch_size, data_type='train')):
                self.last_result = self._chainer.train_on_batch(x, y_true)
                if self.last_result is None:
                    self.last_result = {}
                elif not isinstance(self.last_result, dict):
                    self.last_result = {'loss': self.last_result}
                if 'loss' in self.last_result:
                    self.losses.append(self.last_result.pop('loss'))

                self.train_batches_seen += 1
                self.examples += len(x)

                if self.log_every_n_batches > 0 and self.train_batches_seen % self.log_every_n_batches == 0:
                    self._log(iterator, tensorboard_tag='every_n_batches', tensorboard_index=self.train_batches_seen)

                if self.val_every_n_batches > 0 and self.train_batches_seen % self.val_every_n_batches == 0:
                    self._validate(iterator,
                                   tensorboard_tag='every_n_batches', tensorboard_index=self.train_batches_seen)

                self._send_event(event_name='after_batch')

                if 0 < self.max_batches <= self.train_batches_seen:
                    impatient = True
                    break

                if 0 < self.validation_patience <= self.patience:
                    log.info('Ran out of patience')
                    impatient = True
                    break

            if impatient:
                break

            self.epoch += 1

            if self.log_every_n_epochs > 0 and self.epoch % self.log_every_n_epochs == 0:
                self._log(iterator, tensorboard_tag='every_n_epochs', tensorboard_index=self.epoch)

            if self.val_every_n_epochs > 0 and self.epoch % self.val_every_n_epochs == 0:
                self._validate(iterator, tensorboard_tag='every_n_epochs', tensorboard_index=self.epoch)

            self._send_event(event_name='after_epoch')

            if 0 < self.max_epochs <= self.epoch:
                break

            if 0 < self.validation_patience <= self.patience:
                log.info('Ran out of patience')
                break

    def train(self, iterator: DataLearningIterator) -> None:
        """Call :meth:`~fit_chainer` and then :meth:`~train_on_batches` with provided data iterator as an argument"""
        self.fit_chainer(iterator)
        if callable(getattr(self._chainer, 'train_on_batch', None)):
            try:
                self.train_on_batches(iterator)
            except KeyboardInterrupt:
                log.info('Stopped training')
        else:
            log.warning(f'Using {self.__class__.__name__} for a pipeline without batched training')

        # Run the at-train-exit model-saving logic
        if self.validation_number < 1:
            log.info('Save model to capture early training results')
            self.save()
