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
from typing import Tuple, Dict, Union, Optional, Iterable, Any, Collection

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.trainers.utils import Metric, parse_metrics, prettify_metrics, NumpyArrayEncoder

log = getLogger(__name__)


@register('fit_trainer')
class FitTrainer:
    """
    Trainer class for fitting and evaluating :class:`Estimators <deeppavlov.core.models.estimator.Estimator>`

    Args:
        chainer_config: ``"chainer"`` block of a configuration file
        batch_size: batch_size to use for partial fitting (if available) and evaluation,
            the whole dataset is used if ``batch_size`` is negative or zero (default is ``-1``)
        metrics: iterable of metrics where each metric can be a registered metric name or a dict of ``name`` and
            ``inputs`` where ``name`` is a registered metric name and ``inputs`` is a collection of parameter names
            from chainer’s inner memory that will be passed to the metric function;
            default value for ``inputs`` parameter is a concatenation of chainer’s ``in_y`` and ``out`` fields
            (default is ``('accuracy',)``)
        evaluation_targets: data types on which to evaluate trained pipeline (default is ``('valid', 'test')``)
        show_examples: a flag used to print inputs, expected outputs and predicted outputs for the last batch
            in evaluation logs (default is ``False``)
        tensorboard_log_dir: path to a directory where tensorboard logs can be stored, ignored if None
            (default is ``None``)
        max_test_batches: maximum batches count for pipeline testing and evaluation, ignored if negative
            (default is ``-1``)
        **kwargs: additional parameters whose names will be logged but otherwise ignored
    """

    def __init__(self, chainer_config: dict, *, batch_size: int = -1,
                 metrics: Iterable[Union[str, dict]] = ('accuracy',),
                 evaluation_targets: Iterable[str] = ('valid', 'test'),
                 show_examples: bool = False,
                 tensorboard_log_dir: Optional[Union[str, Path]] = None,
                 max_test_batches: int = -1,
                 **kwargs) -> None:
        if kwargs:
            log.info(f'{self.__class__.__name__} got additional init parameters {list(kwargs)} that will be ignored:')
        self.chainer_config = chainer_config
        self._chainer = Chainer(chainer_config['in'], chainer_config['out'], chainer_config.get('in_y'))
        self.batch_size = batch_size
        self.metrics = parse_metrics(metrics, self._chainer.in_y, self._chainer.out_params)
        self.evaluation_targets = tuple(evaluation_targets)
        self.show_examples = show_examples

        self.max_test_batches = None if max_test_batches < 0 else max_test_batches

        self.tensorboard_log_dir: Optional[Path] = tensorboard_log_dir
        if tensorboard_log_dir is not None:
            try:
                # noinspection PyPackageRequirements
                # noinspection PyUnresolvedReferences
                import tensorflow
            except ImportError:
                log.warning('TensorFlow could not be imported, so tensorboard log directory'
                            f'`{self.tensorboard_log_dir}` will be ignored')
                self.tensorboard_log_dir = None
            else:
                self.tensorboard_log_dir = expand_path(tensorboard_log_dir)
                self._tf = tensorflow

        self._built = False
        self._saved = False
        self._loaded = False

    def fit_chainer(self, iterator: Union[DataFittingIterator, DataLearningIterator]) -> None:
        """
        Build the pipeline :class:`~deeppavlov.core.common.chainer.Chainer` and successively fit
        :class:`Estimator <deeppavlov.core.models.estimator.Estimator>` components using a provided data iterator
        """
        if self._built:
            raise RuntimeError('Cannot fit already built chainer')
        for component_index, component_config in enumerate(self.chainer_config['pipe'], 1):
            component = from_params(component_config, mode='train')
            if 'fit_on' in component_config:
                component: Estimator

                targets = component_config['fit_on']
                if isinstance(targets, str):
                    targets = [targets]

                if self.batch_size > 0 and callable(getattr(component, 'partial_fit', None)):
                    writer = None

                    for i, (x, y) in enumerate(iterator.gen_batches(self.batch_size, shuffle=False)):
                        preprocessed = self._chainer.compute(x, y, targets=targets)
                        # noinspection PyUnresolvedReferences
                        result = component.partial_fit(*preprocessed)

                        if result is not None and self.tensorboard_log_dir is not None:
                            if writer is None:
                                writer = self._tf.summary.FileWriter(str(self.tensorboard_log_dir /
                                                                         f'partial_fit_{component_index}_log'))
                            for name, score in result.items():
                                summary = self._tf.Summary()
                                summary.value.add(tag='partial_fit/' + name, simple_value=score)
                                writer.add_summary(summary, i)
                            writer.flush()
                else:
                    preprocessed = self._chainer.compute(*iterator.get_instances(), targets=targets)
                    if len(targets) == 1:
                        preprocessed = [preprocessed]
                    result: Optional[Dict[str, Iterable[float]]] = component.fit(*preprocessed)

                    if result is not None and self.tensorboard_log_dir is not None:
                        writer = self._tf.summary.FileWriter(str(self.tensorboard_log_dir /
                                                                 f'fit_log_{component_index}'))
                        for name, scores in result.items():
                            for i, score in enumerate(scores):
                                summary = self._tf.Summary()
                                summary.value.add(tag='fit/' + name, simple_value=score)
                                writer.add_summary(summary, i)
                        writer.flush()

                component.save()

            if 'in' in component_config:
                c_in = component_config['in']
                c_out = component_config['out']
                in_y = component_config.get('in_y', None)
                main = component_config.get('main', False)
                self._chainer.append(component, c_in, c_out, in_y, main)
        self._built = True

    def _load(self) -> None:
        if not self._loaded:
            self._chainer.destroy()
            self._chainer = build_model({'chainer': self.chainer_config}, load_trained=self._saved)
            self._loaded = True

    def get_chainer(self) -> Chainer:
        """Returns a :class:`~deeppavlov.core.common.chainer.Chainer` built from ``self.chainer_config`` for inference"""
        self._load()
        return self._chainer

    def train(self, iterator: Union[DataFittingIterator, DataLearningIterator]) -> None:
        """Calls :meth:`~fit_chainer` with provided data iterator as an argument"""
        self.fit_chainer(iterator)
        self._saved = True

    def test(self, data: Iterable[Tuple[Collection[Any], Collection[Any]]],
             metrics: Optional[Collection[Metric]] = None, *,
             start_time: Optional[float] = None, show_examples: Optional[bool] = None) -> dict:
        """
        Calculate metrics and return reports on provided data for currently stored
        :class:`~deeppavlov.core.common.chainer.Chainer`

        Args:
            data: iterable of batches of inputs and expected outputs
            metrics: collection of metrics namedtuples containing names for report, metric functions
                and their inputs names (if omitted, ``self.metrics`` is used)
            start_time: start time for test report
            show_examples: a flag used to return inputs, expected outputs and predicted outputs for the last batch
                in a result report (if omitted, ``self.show_examples`` is used)

        Returns:
            a report dict containing calculated metrics, spent time value, examples count in tested data
            and maybe examples
        """

        if start_time is None:
            start_time = time.time()
        if show_examples is None:
            show_examples = self.show_examples
        if metrics is None:
            metrics = self.metrics

        expected_outputs = list(set().union(self._chainer.out_params, *[m.inputs for m in metrics]))

        outputs = {out: [] for out in expected_outputs}
        examples = 0

        data = islice(data, self.max_test_batches)

        for x, y_true in data:
            examples += len(x)
            y_predicted = list(self._chainer.compute(list(x), list(y_true), targets=expected_outputs))
            if len(expected_outputs) == 1:
                y_predicted = [y_predicted]
            for out, val in zip(outputs.values(), y_predicted):
                out += list(val)

        if examples == 0:
            log.warning('Got empty data iterable for scoring')
            return {'eval_examples_count': 0, 'metrics': None, 'time_spent': str(datetime.timedelta(seconds=0))}

        # metrics_values = [(m.name, m.fn(*[outputs[i] for i in m.inputs])) for m in metrics]
        metrics_values = []
        for metric in metrics:
            value = metric.fn(*[outputs[i] for i in metric.inputs])
            metrics_values.append((metric.alias, value))

        report = {
            'eval_examples_count': examples,
            'metrics': prettify_metrics(metrics_values),
            'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
        }

        if show_examples:
            y_predicted = zip(*[y_predicted_group
                                for out_name, y_predicted_group in zip(expected_outputs, y_predicted)
                                if out_name in self._chainer.out_params])
            if len(self._chainer.out_params) == 1:
                y_predicted = [y_predicted_item[0] for y_predicted_item in y_predicted]
            report['examples'] = [{
                'x': x_item,
                'y_predicted': y_predicted_item,
                'y_true': y_true_item
            } for x_item, y_predicted_item, y_true_item in zip(x, y_predicted, y_true)]

        return report

    def evaluate(self, iterator: DataLearningIterator, evaluation_targets: Optional[Iterable[str]] = None, *,
                 print_reports: bool = True) -> Dict[str, dict]:
        """
        Run :meth:`test` on multiple data types using provided data iterator

        Args:
            iterator: :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for evaluation
            evaluation_targets: iterable of data types to evaluate on
            print_reports: a flag used to print evaluation reports as json lines

        Returns:
            a dictionary with data types as keys and evaluation reports as values
        """
        self._load()
        if evaluation_targets is None:
            evaluation_targets = self.evaluation_targets

        res = {}

        for data_type in evaluation_targets:
            data_gen = iterator.gen_batches(self.batch_size, data_type=data_type, shuffle=False)
            report = self.test(data_gen)
            res[data_type] = report
            if print_reports:
                print(json.dumps({data_type: report}, ensure_ascii=False, cls=NumpyArrayEncoder))

        return res
