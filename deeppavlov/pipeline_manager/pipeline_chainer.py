import datetime
import importlib
import json
import time
from collections import OrderedDict
from typing import List, Callable, Tuple, Dict, Union

from deeppavlov.core.commands.utils import expand_path, set_deeppavlov_root
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.commands.train import _train_batches, _fit_batches, _fit, _test_model
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import model as get_model
from deeppavlov.core.common.metrics_registry import get_metrics_by_names
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.params import from_params
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)
set_deeppavlov_root({})


class Pipeline:
    def __init__(self, pipe: list, iterator: Union[DataLearningIterator, DataFittingIterator],
                 in_x='x', in_y='y', out=None):
        self.pipe = pipe
        self.len = len(pipe)
        self.chainer = Chainer(in_x, in_y, out)
        self.iterator = iterator
        self.elements = []

        self.fill_chainer()

    def fill_chainer(self):
        for x in self.pipe:
            self.append_to_chainer(x)
        return self

    def append_to_chainer(self, element):
        if (not isinstance(element, tuple) and len(element) != 2) or not isinstance(element[1], dict):
            raise ConfigError("")

        component = element[0]
        config = element[1]

        if callable(component):
            component = component(**config)

        self.chainer.append(component, in_x=config.get('in'), out_params=config.get('out'),
                            in_y=config.get('in_y'), main=config.get('main'))
        self.elements.append((component, config))

    def fit_chainer(self, batch_size):
        for component, component_config in self.elements:
            if 'fit_on' in component_config:
                component: Estimator

                preprocessed = self.chainer(*self.iterator.get_instances('train'), to_return=component_config['fit_on'])
                if len(component_config['fit_on']) == 1:
                    preprocessed = [preprocessed]
                else:
                    preprocessed = zip(*preprocessed)
                component.fit(*preprocessed)
                component.save()

            if 'fit_on_batch' in component_config:
                component: Estimator
                component.fit_batches(self.iterator, batch_size)
                component.save()

            if 'in' in component_config:
                c_in = component_config['in']
                c_out = component_config['out']
                in_y = component_config.get('in_y', None)
                main = component_config.get('main', False)
                self.chainer.append(component, c_in, c_out, in_y, main)

        return self.chainer

    # TODO del dependencies from build_model_from_config
    # TODO maybe for this we need refactor _train_batches, _fit_batches, _fit
    def fit(self, metrics: list = ['accuracy'], epochs=100, batch_size=32, validation_patience=5,
            val_every_n_epochs=5, log_every_n_epochs=1, show_examples=False, to_train=True,
            to_validate=True, test_best=True) -> None:

        config_train = {"metrics": metrics,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "validation_patience": validation_patience,
                        "val_every_n_epochs": val_every_n_epochs,
                        "log_every_n_epochs": log_every_n_epochs,
                        "show_examples": show_examples,
                        "to_train": to_train,
                        "to_validate": to_validate,
                        "test_best": test_best}

        train_config = {
            'metrics': ['accuracy'],
            'validate_best': to_validate,
            'test_best': True
        }

        try:
            train_config.update(config_train)
        except KeyError:
            log.warning('Train config is missing. Populating with default values')

        metrics_functions = list(zip(metrics, get_metrics_by_names(metrics)))

        if to_train:
            model = self.fit_chainer(batch_size)

            if callable(getattr(model, 'train_on_batch', None)):
                _train_batches(model, self.iterator, train_config, metrics_functions)
            elif callable(getattr(model, 'fit_batches', None)):
                _fit_batches(model, self.iterator, train_config)
            elif callable(getattr(model, 'fit', None)):
                _fit(model, self.iterator, train_config)
            elif not isinstance(model, Chainer):
                log.warning('Nothing to train')

        if train_config['validate_best'] or train_config['test_best']:
            # try:
            #     model_config['load_path'] = model_config['save_path']
            # except KeyError:
            #     log.warning('No "save_path" parameter for the model, so "load_path" will not be renewed')
            model = build_model_from_config(config, load_trained=True)
            log.info('Testing the best saved model')

            if train_config['validate_best']:
                report = {
                    'valid': _test_model(model, metrics_functions, self.iterator,
                                         train_config.get('batch_size', -1), 'valid')
                }

                print(json.dumps(report, ensure_ascii=False))

            if train_config['test_best']:
                report = {
                    'test': _test_model(model, metrics_functions, self.iterator,
                                        train_config.get('batch_size', -1), 'test')
                }

                print(json.dumps(report, ensure_ascii=False))

        return None

    def load_cheiner(self):



def my_build_model_from_config(config, mode='infer', load_trained=False, as_component=False):
    model_config = config['chainer']

    model = Chainer(model_config['in'], model_config['out'], model_config.get('in_y'), as_component=as_component)

    for component_config in model_config['pipe']:
        if load_trained and ('fit_on' in component_config or 'in_y' in component_config):
            try:
                component_config['load_path'] = component_config['save_path']
            except KeyError:
                log.warning('No "save_path" parameter for the {} component, so "load_path" will not be renewed'
                            .format(component_config.get('name', component_config.get('ref', 'UNKNOWN'))))
        component = from_params(component_config, vocabs=[], mode=mode)

        if 'in' in component_config:
            c_in = component_config['in']
            c_out = component_config['out']
            in_y = component_config.get('in_y', None)
            main = component_config.get('main', False)
            model.append(component, c_in, c_out, in_y, main)

    return model
