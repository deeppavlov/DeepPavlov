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

from pathlib import Path
from typing import Optional

from deeppavlov.core.commands.utils import set_deeppavlov_root, import_packages
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.params import from_params

log = get_logger(__name__)


def build_model_from_config(config: [str, Path, dict], mode: str = 'infer', load_trained: bool = False) -> Chainer:
    """Build and return the model described in corresponding configuration file."""
    if isinstance(config, (str, Path)):
        config = read_json(config)
    set_deeppavlov_root(config)

    import_packages(config.get('metadata', {}).get('imports', []))

    model_config = config['chainer']

    model = Chainer(model_config['in'], model_config['out'], model_config.get('in_y'))

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


def interact_model(config_path: str) -> None:
    """Start interaction with the model described in corresponding configuration file."""
    config = read_json(config_path)
    model = build_model_from_config(config)

    while True:
        args = []
        for in_x in model.in_x:
            args.append([input('{}::'.format(in_x))])
            # check for exit command
            if args[-1][0] in {'exit', 'stop', 'quit', 'q'}:
                return

        pred = model(*args)
        if len(model.out_params) > 1:
            pred = zip(*pred)

        print('>>', *pred)


def predict_on_stream(config_path: str, batch_size: int = 1, file_path: Optional[str] = None) -> None:
    """Make a prediction with the component described in corresponding configuration file."""
    import sys
    import json
    from itertools import islice

    if file_path is None or file_path == '-':
        if sys.stdin.isatty():
            raise RuntimeError('To process data from terminal please use interact mode')
        f = sys.stdin
    else:
        f = open(file_path, encoding='utf8')

    config = read_json(config_path)
    model: Chainer = build_model_from_config(config)

    args_count = len(model.in_x)
    while True:
        batch = list((l.strip() for l in islice(f, batch_size*args_count)))

        if not batch:
            break

        args = []
        for i in range(args_count):
            args.append(batch[i::args_count])

        res = model(*args)
        if len(model.out_params) == 1:
            res = [res]
        for res in zip(*res):
            res = json.dumps(res, ensure_ascii=False)
            print(res, flush=True)

    if f is not sys.stdin:
        f.close()
