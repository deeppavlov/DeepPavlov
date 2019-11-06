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

from logging import getLogger
from pathlib import Path
from typing import Dict, Union, Optional, Iterable

from deeppavlov.core.commands.utils import expand_path, import_packages, parse_config
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import get_model
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.data.utils import get_all_elems_from_json
from deeppavlov.download import deep_download

log = getLogger(__name__)


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


def train_evaluate_model_from_config(config: Union[str, Path, dict],
                                     iterator: Union[DataLearningIterator, DataFittingIterator] = None, *,
                                     to_train: bool = True,
                                     evaluation_targets: Optional[Iterable[str]] = None,
                                     to_validate: Optional[bool] = None,
                                     download: bool = False,
                                     start_epoch_num: Optional[int] = None,
                                     recursive: bool = False) -> Dict[str, Dict[str, float]]:
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

    if 'train' not in config:
        log.warning('Train config is missing. Populating with default values')
    train_config = config.get('train')

    if start_epoch_num is not None:
        train_config['start_epoch_num'] = start_epoch_num

    if 'evaluation_targets' not in train_config and ('validate_best' in train_config
                                                     or 'test_best' in train_config):
        log.warning('"validate_best" and "test_best" parameters are deprecated.'
                    ' Please, use "evaluation_targets" list instead')

        train_config['evaluation_targets'] = []
        if train_config.pop('validate_best', True):
            train_config['evaluation_targets'].append('valid')
        if train_config.pop('test_best', True):
            train_config['evaluation_targets'].append('test')

    trainer_class = get_model(train_config.pop('class_name', 'nn_trainer'))
    trainer = trainer_class(config['chainer'], **train_config)

    if to_train:
        trainer.train(iterator)

    res = {}

    if iterator is not None:
        if to_validate is not None:
            if evaluation_targets is None:
                log.warning('"to_validate" parameter is deprecated and will be removed in future versions.'
                            ' Please, use "evaluation_targets" list instead')
                evaluation_targets = ['test']
                if to_validate:
                    evaluation_targets.append('valid')
            else:
                log.warning('Both "evaluation_targets" and "to_validate" parameters are specified.'
                            ' "to_validate" is deprecated and will be ignored')

        res = trainer.evaluate(iterator, evaluation_targets, print_reports=True)
        trainer.get_chainer().destroy()

    res = {k: v['metrics'] for k, v in res.items()}

    return res
