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

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import REGISTRY

from deeppavlov.core.agent.agent import Agent
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


def build_model_from_config(config, mode='infer'):
    model_config = config['model']
    model_name = model_config['name']

    vocabs = {}
    if 'vocabs' in config:
        for vocab_param_name, vocab_config in config['vocabs'].items():
            vocab_name = vocab_config['name']
            v = from_params(REGISTRY[vocab_name], vocab_config, mode=mode)
            vocabs[vocab_param_name] = v
    model = from_params(REGISTRY[model_name], model_config, vocabs=vocabs, mode=mode)
    model.reset()
    return model


def build_agent_from_config(config_path: str):
    config = read_json(config_path)
    skill_configs = config['skills']
    commutator_config = config['commutator']
    return Agent(skill_configs, commutator_config)


def interact_agent(config_path):
    a = build_agent_from_config(config_path)
    commutator_name = a.commutator_config['name']
    commutator = from_params(REGISTRY[commutator_name], a.commutator_config)

    models = [build_model_from_config(sk) for sk in a.skill_configs]
    while True:
        # get input from user
        context = input(':: ')

        # check for exit command
        if context == 'exit' or context == 'stop' or context == 'quit' or context == 'q':
            return

        predictions = []
        for model in models:
            predictions.append({model.__class__.__name__: model.infer(context, )})
        idx, name, pred = commutator.infer(predictions, )
        print('>>', pred)

        a.history.append({'context': context, "predictions": predictions,
                          "winner": {"idx": idx, "model": name, "prediction": pred}})
        log.debug("Current history: {}".format(a.history))


def interact_model(config_path):
    config = read_json(config_path)
    model = build_model_from_config(config)

    while True:
        # get input from user
        context = input(':: ')

        # check for exit command
        if context == 'exit' or context == 'stop' or context == 'quit' or context == 'q':
            return

        try:
            pred = model.infer(context, )
            print('>>', pred)
        except Exception as e:
            raise e
