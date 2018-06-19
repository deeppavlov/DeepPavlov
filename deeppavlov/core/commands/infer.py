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
from deeppavlov.core.commands.utils import set_deeppavlov_root
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import REGISTRY

from deeppavlov.core.agent.agent import Agent
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


def build_model_from_config(config, mode='infer', load_trained=False, as_component=False):
    set_deeppavlov_root(config)
    model_config = config['chainer']

    model = Chainer(model_config['in'], model_config['out'], model_config.get('in_y'), as_component=as_component)

    for component_config in model_config['pipe']:
        if load_trained and ('fit_on' in component_config or 'in_y' in component_config):
            try:
                component_config['load_path'] = component_config['save_path']
            except KeyError:
                log.warning('No "save_path" parameter for the {} component, so "load_path" will not be renewed'
                            .format(component_config.get('name', component_config.get('ref', 'UNKNOWN'))))
        component = from_params(component_config, mode=mode)

        if 'in' in component_config:
            c_in = component_config['in']
            c_out = component_config['out']
            in_y = component_config.get('in_y', None)
            main = component_config.get('main', False)
            model.append(component, c_in, c_out, in_y, main)

    return model


def build_agent_from_config(config_path: str):
    config = read_json(config_path)
    skill_configs = config['skills']
    commutator_config = config['commutator']
    return Agent(skill_configs, commutator_config)


def interact_agent(config_path):
    a = build_agent_from_config(config_path)
    commutator = from_params(a.commutator_config)

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
        args = []
        for in_x in model.in_x:
            args.append(input('{}::'.format(in_x)))
            # check for exit command
            if args[-1] == 'exit' or args[-1] == 'stop' or args[-1] == 'quit' or args[-1] == 'q':
                return

        if len(args) == 1:
            pred = model(args)
        else:
            pred = model([args])

        print('>>', *pred)


def predict_on_stream(config_path, batch_size=1, file_path=None):
    import sys
    import json
    from itertools import islice

    if file_path is None or file_path == '-':
        if sys.stdin.isatty():
            raise RuntimeError('To process data from terminal please use interact mode')
        f = sys.stdin
    else:
        f = open(file_path)

    config = read_json(config_path)
    model: Chainer = build_model_from_config(config)

    args_count = len(model.in_x)
    while True:
        batch = (l.strip() for l in islice(f, batch_size*args_count))
        if args_count > 1:
            batch = zip(*[batch]*args_count)
        batch = list(batch)

        if not batch:
            break

        for res in model(batch):
            if type(res).__module__ == 'numpy':
                res = res.tolist()
            if not isinstance(res, str):
                res = json.dumps(res, ensure_ascii=False)
            print(res, flush=True)

    if f is not sys.stdin:
        f.close()
