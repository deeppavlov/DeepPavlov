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

import datetime
import json
import time
from collections import OrderedDict
from typing import List, Callable, Tuple

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import model as get_model
from deeppavlov.core.common.metrics_registry import get_metrics_by_names
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.dataset import Dataset
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


def _fit(model: Estimator, dataset: Dataset, train_config={}):
    x, y = dataset.iter_all('train')
    model.fit(x, y)
    model.save()
    return model


def train_chainer(config_path: str):
    config = read_json(config_path)

    reader_config = config['dataset_reader']
    reader = get_model(reader_config['name'])()
    data_path = expand_path(reader_config.get('data_path', ''))
    data = reader.read(data_path)

    dataset_config = config['dataset']
    dataset: Dataset = from_params(dataset_config, data=data)
    *x, y = dataset.iter_all('train')

    chainer_config: dict = config['chainer']
    chainer_forward = Chainer(chainer_config['in'], chainer_config['out'])
    chainer_train = Chainer(chainer_config['in_train'],
                            chainer_config.get('out_train', chainer_config['out']))
    forward_map = set(chainer_config['in'])
    train_map = set(chainer_config['in_train'])
    for component_config in chainer_config['pipe']:
        component = from_params(component_config, vocabs=[], mode='train')
        if 'fit_on' in component_config:
            component: Estimator

            component.fit(*chainer_train(*x, y, to_return=component_config['fit_on']))
            component.save()

        if 'in' in component_config:
            c_in = component_config.pop('in')
            c_out = component_config.pop('out')
            if train_map.issuperset(c_in):
                chainer_train.append(c_in, c_out, component)
                train_map = train_map.union(c_out)
            if forward_map.issuperset(c_in):
                chainer_forward.append(c_in, c_out, component)
                forward_map = forward_map.union(c_out)
    print(chainer_forward(['Helllo tqhere', 'firend']))
    raise Exception


def train_model_from_config(config_path: str):
    config = read_json(config_path)
    if 'chainer' in config:
        return train_chainer(config_path)

    reader_config = config['dataset_reader']
    reader = get_model(reader_config['name'])()
    data_path = expand_path(reader_config.get('data_path', ''))
    data = reader.read(data_path)

    dataset_config = config['dataset']
    dataset: Dataset = from_params(dataset_config, data=data)

    vocabs = {}
    for vocab_param_name, vocab_config in config.get('vocabs', {}).items():
        v: Estimator = from_params(vocab_config, mode='train')
        vocabs[vocab_param_name] = _fit(v, dataset)

    model_config = config['model']
    model = from_params(model_config, vocabs=vocabs, mode='train')

    train_config = {
        'metrics': ['accuracy'],

        'validate_best': True,
        'test_best': True
    }

    try:
        train_config.update(config['train'])
    except KeyError:
        log.warning('Train config is missing. Populating with default values')

    metrics_functions = list(zip(train_config['metrics'], get_metrics_by_names(train_config['metrics'])))

    if callable(getattr(model, 'train_on_batch', None)):
        _train_batches(model, dataset, train_config, metrics_functions)
    _fit(model, dataset, train_config)

    if train_config['validate_best'] or train_config['test_best']:
        try:
            model_config['load_path'] = model_config['save_path']
        except KeyError:
            log.warning('No "save_path" parameter for the model, so "load_path" will not be renewed')
        model = from_params(model_config, vocabs=vocabs, mode='infer')
        log.info('Testing the best saved model')

        if train_config['validate_best']:
            report = {
                'valid': _test_model(model, metrics_functions, dataset, train_config.get('batch_size', -1), 'valid')
            }

            print(json.dumps(report, ensure_ascii=False))

        if train_config['test_best']:
            report = {
                'test': _test_model(model, metrics_functions, dataset, train_config.get('batch_size', -1), 'test')
            }

            print(json.dumps(report, ensure_ascii=False))


def _test_model(model: Component, metrics_functions: List[Tuple[str, Callable]],
                dataset: Dataset, batch_size=-1, data_type='valid', start_time=None):
    if start_time is None:
        start_time = time.time()

    val_y_true = []
    val_y_predicted = []
    for x, y_true in dataset.batch_generator(batch_size, data_type, shuffle=False):
        y_predicted = list(model(list(x)))
        val_y_true += y_true
        val_y_predicted += y_predicted

    metrics = [(s, f(val_y_true, val_y_predicted)) for s, f in metrics_functions]

    report = {
        'examples_seen': len(val_y_true),
        'metrics': OrderedDict(metrics),
        'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time)))
    }
    return report


def _train_batches(model: NNModel, dataset: Dataset, train_config: dict,
                   metrics_functions: List[Tuple[str, Callable]]):

    default_train_config = {
        'epochs': 0,
        'batch_size': 1,

        'metric_optimization': 'maximize',

        'validation_patience': 5,
        'val_every_n_epochs': 0,

        'log_every_n_batches': 0,
        'log_every_n_epochs': 0,
        # 'show_examples': False,

        'validate_best': True,
        'test_best': True
    }

    train_config = dict(default_train_config, ** train_config)

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
    epochs = 0
    examples = 0
    saved = False
    patience = 0
    log_on = train_config['log_every_n_batches'] > 0 or train_config['log_every_n_epochs'] > 0
    train_y_true = []
    train_y_predicted = []
    start_time = time.time()
    try:
        while True:
            for x, y_true in dataset.batch_generator(train_config['batch_size']):
                if log_on:
                    y_predicted = list(model(list(x)))
                    train_y_true += y_true
                    train_y_predicted += y_predicted
                model.train_on_batch(x, y_true)
                i += 1
                examples += len(x)

                if train_config['log_every_n_batches'] > 0 and i % train_config['log_every_n_batches'] == 0:
                    metrics = [(s, f(train_y_true, train_y_predicted)) for s, f in metrics_functions]
                    report = {
                        'epochs_done': epochs,
                        'batches_seen': i,
                        'examples_seen': examples,
                        'metrics': dict(metrics),
                        'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time)))
                    }
                    report = {'train': report}
                    print(json.dumps(report, ensure_ascii=False))
                    train_y_true = []
                    train_y_predicted = []

            epochs += 1

            if train_config['log_every_n_epochs'] > 0 and epochs % train_config['log_every_n_epochs'] == 0\
                    and train_y_true:
                metrics = [(s, f(train_y_true, train_y_predicted)) for s, f in metrics_functions]
                report = {
                    'epochs_done': epochs,
                    'batches_seen': i,
                    'examples_seen': examples,
                    'metrics': dict(metrics),
                    'time_spent': str(datetime.timedelta(seconds=round(time.time() - start_time)))
                }
                report = {'train': report}
                print(json.dumps(report, ensure_ascii=False))
                train_y_true = []
                train_y_predicted = []

            if train_config['val_every_n_epochs'] > 0 and epochs % train_config['val_every_n_epochs'] == 0:
                report = _test_model(model, metrics_functions, dataset, train_config['batch_size'], 'valid', start_time)

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
