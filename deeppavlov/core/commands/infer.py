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
import json
import pickle
import sys
from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

from deeppavlov.core.commands.utils import import_packages, parse_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.download import deep_download

log = getLogger(__name__)


def build_model(config: Union[str, Path, dict], mode: str = 'infer',
                load_trained: bool = False, download: bool = False,
                serialized: Optional[bytes] = None) -> Chainer:
    """Build and return the model described in corresponding configuration file."""
    config = parse_config(config)

    if serialized:
        serialized: list = pickle.loads(serialized)

    if download:
        deep_download(config)

    import_packages(config.get('metadata', {}).get('imports', []))

    model_config = config['chainer']

    model = Chainer(model_config['in'], model_config['out'], model_config.get('in_y'))

    for component_config in model_config['pipe']:
        if load_trained and ('fit_on' in component_config or 'in_y' in component_config):
            try:
                component_config['load_path'] = component_config['save_path']
            except KeyError:
                log.warning('No "save_path" parameter for the {} component, so "load_path" will not be renewed'
                            .format(component_config.get('class_name', component_config.get('ref', 'UNKNOWN'))))

        if serialized and 'in' in component_config:
            component_serialized = serialized.pop(0)
        else:
            component_serialized = None

        component = from_params(component_config, mode=mode, serialized=component_serialized)

        if 'id' in component_config:
            model._components_dict[component_config['id']] = component

        if 'in' in component_config:
            c_in = component_config['in']
            c_out = component_config['out']
            in_y = component_config.get('in_y', None)
            main = component_config.get('main', False)
            model.append(component, c_in, c_out, in_y, main)

    return model


def interact_model(config: Union[str, Path, dict]) -> None:
    """Start interaction with the model described in corresponding configuration file."""
    model = build_model(config)

    while True:
        args = []
        for in_x in model.in_x:
            args.append((input('{}::'.format(in_x)),))
            # check for exit command
            if args[-1][0] in {'exit', 'stop', 'quit', 'q'}:
                return

        pred = model(*args)
        if len(model.out_params) > 1:
            pred = zip(*pred)

        print('>>', *pred)


def predict_on_stream(config: Union[str, Path, dict],
                      batch_size: Optional[int] = None,
                      file_path: Optional[str] = None) -> None:
    """Make a prediction with the component described in corresponding configuration file."""

    batch_size = batch_size or 1
    if file_path is None or file_path == '-':
        if sys.stdin.isatty():
            raise RuntimeError('To process data from terminal please use interact mode')
        f = sys.stdin
    else:
        f = open(file_path, encoding='utf8')

    model: Chainer = build_model(config)

    args_count = len(model.in_x)
    while True:
        batch = list((l.strip() for l in islice(f, batch_size * args_count)))

        if not batch:
            break

        args = []
        for i in range(args_count):
            args.append(batch[i::args_count])

        res = model(*args)
        if len(model.out_params) == 1:
            res = [res]
        for res in zip(*res):
            res = json.dumps(jsonify_data(res), ensure_ascii=False)
            print(res, flush=True)

    if f is not sys.stdin:
        f.close()
