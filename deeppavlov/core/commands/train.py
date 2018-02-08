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

import sys

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import REGISTRY
from deeppavlov.core.commands.infer import build_agent_from_config
from deeppavlov.core.common.params import from_params
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.common import paths
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)

# TODO pass paths to local model configs to agent config.


def train_agent_models(config_path: str):
    usr_dir = paths.USR_PATH
    a = build_agent_from_config(config_path)

    for skill_config in a.skill_configs:
        model_config = skill_config['model']
        model_name = model_config['name']

        if issubclass(REGISTRY[model_name], Trainable):
            reader_config = skill_config['dataset_reader']
            reader = from_params(REGISTRY[reader_config['name']], {})
            data = reader.read(reader_config.get('data_path', usr_dir))

            dataset_config = skill_config['dataset']
            dataset_name = dataset_config['name']
            dataset = from_params(REGISTRY[dataset_name], dataset_config, data=data)

            model = from_params(REGISTRY[model_name], model_config)
            model.train(dataset)
        else:
            log.warning('Model {} is not an instance of Trainable, skip training.'.format(model_name))


def train_model_from_config(config_path: str, mode='train'):
    usr_dir = paths.USR_PATH
    config = read_json(config_path)

    reader_config = config['dataset_reader']
    # NOTE: Why there are no params for dataset reader? Because doesn't have __init__()
    reader = from_params(REGISTRY[reader_config['name']], {})
    data = reader.read(reader_config.get('data_path', usr_dir))

    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    dataset = from_params(REGISTRY[dataset_name], dataset_config, data=data)

    vocabs = {}
    if 'vocabs' in config:
        for vocab_param_name, vocab_config in config['vocabs'].items():
            vocab_name = vocab_config['name']
            v = from_params(REGISTRY[vocab_name], vocab_config, mode=mode)
            v.train(dataset.iter_all('train'))
            vocabs[vocab_param_name] = v

    model_config = config['model']
    model_name = model_config['name']
    model = from_params(REGISTRY[model_name], model_config, vocabs=vocabs, mode=mode)

    model.train(dataset)

    # The result is a saved to user_dir trained model.
