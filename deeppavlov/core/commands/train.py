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

import datetime
import json
import time
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import expand_path, import_packages, parse_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.metrics_registry import get_metric_by_name
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import get_model
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.data.utils import get_all_elems_from_json
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.download import deep_download

log = get_logger(__name__)

Metric = namedtuple('Metric', ['name', 'fn', 'inputs'])


def _parse_metrics(metrics: List[Union[str, dict]], in_y: List[str], out_vars: List[str]) -> List[Metric]:
    metrics_functions = []
    for metric in metrics:
        if isinstance(metric, str):
            metric = {'name': metric}

        metric_name = metric['name']

        f = get_metric_by_name(metric_name)

        inputs = metric.get('inputs', in_y + out_vars)
        if isinstance(inputs, str):
            inputs = [inputs]

        metrics_functions.append(Metric(metric_name, f, inputs))
    return metrics_functions


def prettify_metrics(metrics: List[Tuple[str, float]], precision: int = 4) -> OrderedDict:
    """Prettifies the dictionary of metrics."""
    prettified_metrics = OrderedDict()
    for key, value in metrics:
        value = round(value, precision)
        prettified_metrics[key] = value
    return prettified_metrics


def fit_chainer(config: dict, iterator: Union[DataLearningIterator, DataFittingIterator]) -> Chainer:
    """Fit and return the chainer described in corresponding configuration dictionary."""
    chainer_config: dict = config['chainer']
    chainer = Chainer(chainer_config['in'], chainer_config['out'], chainer_config.get('in_y'))
    for component_config in chainer_config['pipe']:
        component = from_params(component_config, mode='train')
        if ('fit_on' in component_config) and \
                (not callable(getattr(component, 'partial_fit', None))):
            component: Estimator

            targets = component_config['fit_on']
            if isinstance(targets, str):
                targets = [targets]

            preprocessed = chainer.compute(*iterator.get_instances('train'), targets=targets)
            if len(component_config['fit_on']) == 1:
                preprocessed = [preprocessed]

            result = component.fit(*preprocessed)
            if result is not None and config['train'].get('tensorboard_log_dir') is not None:
                import tensorflow as tf
                tb_log_dir = expand_path(config['train']['tensorboard_log_dir'])
                writer = tf.summary.FileWriter(str(tb_log_dir / 'fit_log'))

                for name, scores in result.items():
                    for i, score in enumerate(scores):
                        summ = tf.Summary()
                        summ.value.add(tag='fit/' + name, simple_value=score)
                        writer.add_summary(summ, i)
                writer.flush()

            component.save()

        if 'fit_on_batch' in component_config:
            log.warning('`fit_on_batch` is deprecated and will be removed in future versions.'
                        ' Please use `fit_on` instead.')
        if ('fit_on_batch' in component_config) or \
                (('fit_on' in component_config) and
                 callable(getattr(component, 'partial_fit', None))):
            component: Estimator
            try:
                targets = component_config['fit_on']
            except KeyError:
                targets = component_config['fit_on_batch']
            if isinstance(targets, str):
                targets = [targets]

            for i, data in enumerate(iterator.gen_batches(config['train']['batch_size'], shuffle=False)):
                preprocessed = chainer.compute(*data, targets=targets)
                if len(targets) == 1:
                    preprocessed = [preprocessed]
                result = component.partial_fit(*preprocessed)

                if result is not None and config['train'].get('tensorboard_log_dir') is not None:
                    if i == 0:
                        import tensorflow as tf
                        tb_log_dir = expand_path(config['train']['tensorboard_log_dir'])
                        writer = tf.summary.FileWriter(str(tb_log_dir / 'fit_batches_log'))

                    for name, score in result.items():
                        summ = tf.Summary()
                        summ.value.add(tag='fit_batches/' + name, simple_value=score)
                        writer.add_summary(summ, i)
                    writer.flush()

            component.save()

        if 'in' in component_config:
            c_in = component_config['in']
            c_out = component_config['out']
            in_y = component_config.get('in_y', None)
            main = component_config.get('main', False)
            chainer.append(component, c_in, c_out, in_y, main)
    return chainer


def read_data_by_config(config: dict):
    """Read data by dataset_reader from specified config."""
    dataset_config = config.get('dataset', None)

    if dataset_config:
        config.pop('dataset')
        ds_type = dataset_config['type']
        if ds_type == 'classification':
            reader = {'class_name': 'basic_classification_reader'}
            iterator = {'class_name': 'basic_classification_iterator'}
            config['dataset_reader'] = {**dataset_config, **reader}
            config['dataset_iterator'] = {**dataset_config, **iterator}
        else:
            raise Exception("Unsupported dataset type: {}".format(ds_type))

    try:
        reader_config = dict(config['dataset_reader'])
    except KeyError:
        raise ConfigError("No dataset reader is provided in the JSON config.")

    reader = get_model(reader_config.pop('class_name'))()
    data_path = reader_config.pop('data_path', '')
    if isinstance(data_path, list):
        data_path = [expand_path(x) for x in data_path]
    else:
        data_path = expand_path(data_path)

    return reader.read(data_path, **reader_config)


def get_iterator_from_config(config: dict, data: dict):
    """Create iterator (from config) for specified data."""
    iterator_config = config['dataset_iterator']
    iterator: Union[DataLearningIterator, DataFittingIterator] = from_params(iterator_config,
                                                                             data=data)
    return iterator


def train_evaluate_model_from_config(config: [str, Path, dict], iterator=None, *,
                                     to_train=True, to_validate=True, download=False,
                                     start_epoch_num=0, recursive=False) -> Dict[str, Dict[str, float]]:
    """Make training and evaluation of the model described in corresponding configuration file."""
    config = parse_config(config)

    if download:
        deep_download(config)

    if to_train and recursive:
        for subconfig in get_all_elems_from_json(config['chainer'], 'config_path'):
            log.info(f'Training "{subconfig}"')
            train_evaluate_model_from_config(subconfig, download=False, recursive=True)

    import_packages(config.get('metadata', {}).get('imports', []))

    if iterator is None:
        try:
            data = read_data_by_config(config)
        except ConfigError as e:
            to_train = False
            log.warning(f'Skipping training. {e.message}')
        else:
            iterator = get_iterator_from_config(config, data)

    train_config = {
        'metrics': ['accuracy'],
        'validate_best': to_validate,
        'test_best': True,
        'show_examples': False
    }

    try:
        train_config.update(config['train'])
    except KeyError:
        log.warning('Train config is missing. Populating with default values')

    in_y = config['chainer'].get('in_y', ['y'])
    if isinstance(in_y, str):
        in_y = [in_y]
    if isinstance(config['chainer']['out'], str):
        config['chainer']['out'] = [config['chainer']['out']]
    metrics_functions = _parse_metrics(train_config['metrics'], in_y, config['chainer']['out'])

    if to_train:
        model = fit_chainer(config, iterator)

        if callable(getattr(model, 'train_on_batch', None)):
            _train_batches(model, iterator, train_config, metrics_functions, start_epoch_num=start_epoch_num)

        model.destroy()

    res = {}

    if iterator is not None and (train_config['validate_best'] or train_config['test_best']):
        model = build_model(config, load_trained=to_train)
        log.info('Testing the best saved model')

        if train_config['validate_best']:
            report = {
                'valid': _test_model(model, metrics_functions, iterator,
                                     train_config.get('batch_size', -1), 'valid',
                                     show_examples=train_config['show_examples'])
            }

            res['valid'] = report['valid']['metrics']

            print(json.dumps(report, ensure_ascii=False))

        if train_config['test_best']:
            report = {
                'test': _test_model(model, metrics_functions, iterator,
                                    train_config.get('batch_size', -1), 'test',
                                    show_examples=train_config['show_examples'])
            }

            res['test'] = report['test']['metrics']

            print(json.dumps(report, ensure_ascii=False))

        model.destroy()

    return res


def _test_model(model: Chainer, metrics_functions: List[Metric],
                iterator: DataLearningIterator, batch_size=-1, data_type='valid',
                start_time: float=None, show_examples=False) -> Dict[str, Union[int, OrderedDict, str]]:
    if start_time is None:
        start_time = time.time()

    expected_outputs = list(set().union(model.out_params, *[m.inputs for m in metrics_functions]))

    if not iterator.data[data_type]:
        log.warning(f'Could not log examples for {data_type}, assuming it\'s empty')
        return {'eval_examples_count': 0, 'metrics': None, 'time_spent': str(datetime.timedelta(seconds=0))}

    outputs = {out: [] for out in expected_outputs}
    examples = 0
    for x, y_true in iterator.gen_batches(batch_size, data_type, shuffle=False):
        examples += len(x)
        y_predicted = list(model.compute(list(x), list(y_true), targets=expected_outputs))
        if len(expected_outputs) == 1:
            y_predicted = [y_predicted]
        for out, val in zip(outputs.values(), y_predicted):
            out += list(val)

    metrics = [(m.name, m.fn(*[outputs[i] for i in m.inputs])) for m in metrics_functions]

    report = {
        'eval_examples_count': examples,
        'metrics': prettify_metrics(metrics),
        'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
    }

    if show_examples:
        try:
            y_predicted = zip(*[y_predicted_group
                                for out_name, y_predicted_group in zip(expected_outputs, y_predicted)
                                if out_name in model.out_params])
            if len(model.out_params) == 1:
                y_predicted = [y_predicted_item[0] for y_predicted_item in y_predicted]
            report['examples'] = [{
                'x': x_item,
                'y_predicted': y_predicted_item,
                'y_true': y_true_item
            } for x_item, y_predicted_item, y_true_item in zip(x, y_predicted, y_true)]
        except NameError:
            log.warning(f'Could not log examples for {data_type}, assuming it\'s empty')

    return report


def _train_batches(model: Chainer, iterator: DataLearningIterator, train_config: dict,
                   metrics_functions: List[Metric], *, start_epoch_num: Optional[int] = None) -> NNModel:

    default_train_config = {
        'epochs': 0,
        'start_epoch_num': 0,
        'max_batches': 0,
        'batch_size': 1,

        'metric_optimization': 'maximize',

        'validation_patience': 5,
        'val_every_n_epochs': 0,
        'val_every_n_batches': 0,

        'log_every_n_batches': 0,
        'log_every_n_epochs': 0,

        'validate_best': True,
        'test_best': True,
        'tensorboard_log_dir': None,
    }

    train_config = dict(default_train_config, **train_config)

    if 'train_metrics' in train_config:
        train_metrics_functions = _parse_metrics(train_config['train_metrics'], model.in_y, model.out_params)
    else:
        train_metrics_functions = metrics_functions
    expected_outputs = list(set().union(model.out_params, *[m.inputs for m in train_metrics_functions]))

    if train_config['metric_optimization'] == 'maximize':
        def improved(score, best):
            return score > best
        best = float('-inf')
    elif train_config['metric_optimization'] == 'minimize':
        def improved(score, best):
            return score < best
        best = float('inf')
    else:
        raise ConfigError('metric_optimization has to be one of {}'.format(['maximize', 'minimize']))

    i = 0
    epochs = start_epoch_num if start_epoch_num is not None else train_config['start_epoch_num']
    examples = 0
    saved = False
    patience = 0
    log_on = train_config['log_every_n_batches'] > 0 or train_config['log_every_n_epochs'] > 0
    outputs = {key: [] for key in expected_outputs}
    losses = []
    start_time = time.time()
    break_flag = False

    if train_config['tensorboard_log_dir'] is not None:
        import tensorflow as tf
        tb_log_dir = expand_path(train_config['tensorboard_log_dir'])

        tb_train_writer = tf.summary.FileWriter(str(tb_log_dir / 'train_log'))
        tb_valid_writer = tf.summary.FileWriter(str(tb_log_dir / 'valid_log'))

    # validate first (important if model is pre-trained)
    if train_config['val_every_n_epochs'] > 0 or train_config['val_every_n_batches'] > 0:
        report = _test_model(model, metrics_functions, iterator,
                             train_config['batch_size'], 'valid', start_time, train_config['show_examples'])
        report['epochs_done'] = epochs
        report['batches_seen'] = i
        report['train_examples_seen'] = examples

        metrics = list(report['metrics'].items())

        m_name, score = metrics[0]
        if improved(score, best):
            patience = 0
            log.info('New best {} of {}'.format(m_name, score))
            best = score
            log.info('Saving model')
            model.save()
            saved = True
        else:
            patience += 1
            log.info('Did not improve on the {} of {}'.format(m_name, best))

        report['impatience'] = patience
        if train_config['validation_patience'] > 0:
            report['patience_limit'] = train_config['validation_patience']

        model.process_event(event_name='after_validation', data=report)
        report = {'valid': report}
        print(json.dumps(report, ensure_ascii=False))

    try:
        while True:
            for x, y_true in iterator.gen_batches(train_config['batch_size']):
                if log_on and len(train_metrics_functions) > 0:
                    y_predicted = list(model.compute(list(x), list(y_true), targets=expected_outputs))
                    if len(expected_outputs) == 1:
                        y_predicted = [y_predicted]
                    for out, val in zip(outputs.values(), y_predicted):
                        out += list(val)
                result = model.train_on_batch(x, y_true)
                if not isinstance(result, dict):
                    result = {'loss': result} if result is not None else {}
                if 'loss' in result:
                    losses.append(result['loss'])
                i += 1
                examples += len(x)

                if train_config['log_every_n_batches'] > 0 and i % train_config['log_every_n_batches'] == 0:
                    metrics = [(m.name, m.fn(*[outputs[i] for i in m.inputs])) for m in train_metrics_functions]
                    report = {
                        'epochs_done': epochs,
                        'batches_seen': i,
                        'examples_seen': examples,
                        'metrics': prettify_metrics(metrics),
                        'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
                    }
                    default_report_keys = list(report.keys())
                    report.update(result)

                    if train_config['show_examples']:
                        try:
                            y_predicted = zip(*[y_predicted_group
                                                for out_name, y_predicted_group in zip(expected_outputs, y_predicted)
                                                if out_name in model.out_params])
                            if len(model.out_params) == 1:
                                y_predicted = [y_predicted_item[0] for y_predicted_item in y_predicted]
                            report['examples'] = [{
                                'x': x_item,
                                'y_predicted': y_predicted_item,
                                'y_true': y_true_item
                            } for x_item, y_predicted_item, y_true_item
                                in zip(x, y_predicted, y_true)]
                        except NameError:
                            log.warning('Could not log examples as y_predicted is not defined')

                    if losses:
                        report['loss'] = sum(losses)/len(losses)
                        losses = []

                    model.process_event(event_name='after_train_log', data=report)

                    if train_config['tensorboard_log_dir'] is not None:
                        summ = tf.Summary()

                        for name, score in metrics:
                            summ.value.add(tag='every_n_batches/' + name, simple_value=score)
                        for name, score in report.items():
                            if name not in default_report_keys:
                                summ.value.add(tag='every_n_batches/' + name, simple_value=score)

                        tb_train_writer.add_summary(summ, i)
                        tb_train_writer.flush()

                    report = {'train': report}
                    print(json.dumps(report, ensure_ascii=False))
                    for out in outputs.values():
                        out.clear()

                if train_config['val_every_n_batches'] > 0 and i % train_config['val_every_n_batches'] == 0:
                    report = _test_model(model, metrics_functions, iterator,
                                         train_config['batch_size'], 'valid', start_time, train_config['show_examples'])
                    report['epochs_done'] = epochs
                    report['batches_seen'] = i
                    report['train_examples_seen'] = examples

                    metrics = list(report['metrics'].items())

                    if train_config['tensorboard_log_dir'] is not None:
                        summ = tf.Summary()
                        for name, score in metrics:
                            summ.value.add(tag='every_n_batches/' + name, simple_value=score)
                        tb_valid_writer.add_summary(summ, i)
                        tb_valid_writer.flush()


                    m_name, score = metrics[0]
                    if improved(score, best):
                        patience = 0
                        log.info('New best {} of {}'.format(m_name, score))
                        best = score
                        log.info('Saving model')
                        model.save()
                        saved = True
                    else:
                        patience += 1
                        log.info('Did not improve on the {} of {}'.format(m_name, best))

                    report['impatience'] = patience
                    if train_config['validation_patience'] > 0:
                        report['patience_limit'] = train_config['validation_patience']

                    model.process_event(event_name='after_validation', data=report)
                    report = {'valid': report}
                    print(json.dumps(report, ensure_ascii=False))

                    if patience >= train_config['validation_patience'] > 0:
                        log.info('Ran out of patience')
                        break_flag = True
                        break

                if i >= train_config['max_batches'] > 0:
                    break_flag = True
                    break

                report = {
                    'epochs_done': epochs,
                    'batches_seen': i,
                    'train_examples_seen': examples,
                    'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
                }
                model.process_event(event_name='after_batch', data=report)
            if break_flag:
                break

            epochs += 1

            report = {
                'epochs_done': epochs,
                'batches_seen': i,
                'train_examples_seen': examples,
                'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
            }
            model.process_event(event_name='after_epoch', data=report)

            if train_config['log_every_n_epochs'] > 0 and epochs % train_config['log_every_n_epochs'] == 0\
                    and outputs:
                metrics = [(m.name, m.fn(*[outputs[i] for i in m.inputs])) for m in train_metrics_functions]
                report = {
                    'epochs_done': epochs,
                    'batches_seen': i,
                    'train_examples_seen': examples,
                    'metrics': prettify_metrics(metrics),
                    'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time + 0.5)))
                }
                default_report_keys = list(report.keys())
                report.update(result)

                if train_config['show_examples']:
                    try:
                        y_predicted = zip(*[y_predicted_group
                                            for out_name, y_predicted_group in zip(expected_outputs, y_predicted)
                                            if out_name in model.out_params])
                        if len(model.out_params) == 1:
                            y_predicted = [y_predicted_item[0] for y_predicted_item in y_predicted]
                        report['examples'] = [{
                            'x': x_item,
                            'y_predicted': y_predicted_item,
                            'y_true': y_true_item
                        } for x_item, y_predicted_item, y_true_item
                            in zip(x, y_predicted, y_true)]
                    except NameError:
                        log.warning('Could not log examples')

                if losses:
                    report['loss'] = sum(losses)/len(losses)
                    losses = []

                model.process_event(event_name='after_train_log', data=report)

                if train_config['tensorboard_log_dir'] is not None:
                    summ = tf.Summary()

                    for name, score in metrics:
                        summ.value.add(tag='every_n_epochs/' + name, simple_value=score)
                    for name, score in report.items():
                        if name not in default_report_keys:
                            summ.value.add(tag='every_n_epochs/' + name, simple_value=score)

                    tb_train_writer.add_summary(summ, epochs)
                    tb_train_writer.flush()

                report = {'train': report}
                print(json.dumps(report, ensure_ascii=False))
                for out in outputs.values():
                    out.clear()

            if train_config['val_every_n_epochs'] > 0 and epochs % train_config['val_every_n_epochs'] == 0:
                report = _test_model(model, metrics_functions, iterator,
                                     train_config['batch_size'], 'valid', start_time, train_config['show_examples'])
                report['epochs_done'] = epochs
                report['batches_seen'] = i
                report['train_examples_seen'] = examples

                metrics = list(report['metrics'].items())

                if train_config['tensorboard_log_dir'] is not None:
                    summ = tf.Summary()
                    for name, score in metrics:
                        summ.value.add(tag='every_n_epochs/' + name, simple_value=score)
                    tb_valid_writer.add_summary(summ, epochs)
                    tb_valid_writer.flush()

                m_name, score = metrics[0]
                if improved(score, best):
                    patience = 0
                    log.info('New best {} of {}'.format(m_name, score))
                    best = score
                    log.info('Saving model')
                    model.save()
                    saved = True
                else:
                    patience += 1
                    log.info('Did not improve on the {} of {}'.format(m_name, best))

                report['impatience'] = patience
                if train_config['validation_patience'] > 0:
                    report['patience_limit'] = train_config['validation_patience']

                model.process_event(event_name='after_validation', data=report)
                report = {'valid': report}
                print(json.dumps(report, ensure_ascii=False))

                if patience >= train_config['validation_patience'] > 0:
                    log.info('Ran out of patience')
                    break

            if epochs >= train_config['epochs'] > 0:
                break
    except KeyboardInterrupt:
        log.info('Stopped training')

    if not saved:
        log.info('Saving model')
        model.save()

    return model
