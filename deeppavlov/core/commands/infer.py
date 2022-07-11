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
from collections import defaultdict
from itertools import islice
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm import tqdm

from deeppavlov.core.commands.train import read_data_by_config, get_iterator_from_config
from deeppavlov.core.commands.utils import import_packages, parse_config, expand_path
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.file import save_jsonl
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.utils import jsonify_data
from deeppavlov.download import deep_download

log = getLogger(__name__)

SUPER_GLUE_TASKS = {
    'boolq': 'BoolQ',
    'copa': 'COPA',
    'danetqa': 'DaNetQA',
    'lidirus': 'LiDiRus',
    'multirc': 'MultiRC',
    'muserc': 'MuSeRC',
    'parus': 'PARus',
    'rcb': 'RCB',
    'record': 'ReCoRD',
    'rucos': 'RuCoS',
    'russe': 'RUSSE',
    'rwsd': 'RWSD',
    'terra': 'TERRa'
}


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


def submit(config: Union[str, Path, dict], output_path: Optional[Union[str, Path]] = None) -> None:
    """Creates submission file for the Russian SuperGLUE task. Supported tasks list will be extended in the future.

    Args:
        config: Configuration of the model.
        output_path: Path to output file. If None, file name is selected according corresponding task name.

    """

    config = parse_config(config)
    data = read_data_by_config(config)
    iterator = get_iterator_from_config(config, data)
    task_name = config['dataset_reader']['name']

    data_gen = iterator.gen_batches(1, data_type='test', shuffle=False)

    model = build_model(config)

    submission = []

    if task_name in {'record', 'rucos'}:
        output = defaultdict(
            lambda: {
                'predicted': [],
                'probability': []
            }
        )

        for x, _ in tqdm(data_gen):
            indices, _, _, entities, _ = x[0]
            prediction = model.compute(x)[:, 1]
            output[indices]['predicted'].append(entities)
            output[indices]['probability'].append(prediction)

        for key, value in output.items():
            answer_index = np.argmax(value['probability'])
            answer = value['predicted'][answer_index]
            submission.append({'idx': int(key.split('-')[1]), 'label': answer})

    elif task_name in {'copa', 'parus'}:
        for idx, (x, _) in enumerate(tqdm(data_gen)):
            prediction = model.compute(x)[0]
            label = int(prediction == 'choice2')
            submission.append({'idx': idx, 'label': label})

    elif task_name in {'muserc', 'multirc'}:
        output = {}
        for x, _ in tqdm(data_gen):
            contexts, answers, indices = x[0]

            prediction = model(contexts, answers)

            paragraph_idx = indices['paragraph']
            question_idx = indices['question']
            answer_idx = indices['answer']

            label = int(prediction[0] == 'True')
            if paragraph_idx not in output:
                output[paragraph_idx] = {
                    'idx': paragraph_idx,
                    'passage': {
                        'questions': [
                            {
                                'idx': question_idx,
                                'answers': [{'idx': answer_idx, 'label': label}]
                            }
                        ]
                    }
                }
    
            questions = output[paragraph_idx]['passage']['questions']
            question_indices = set(el['idx'] for el in questions)
            if question_idx not in question_indices:
                output[paragraph_idx]['passage']['questions'].append({
                    'idx': question_idx,
                    'answers': [{'idx': answer_idx, 'label': label}]
                })
            else:
                for question in questions:
                    if question['idx'] == question_idx:
                        question['answers'].append({'idx': answer_idx, 'label': label})
    
        submission = list(output.values())

    elif task_name in SUPER_GLUE_TASKS:
        for idx, (x, _) in enumerate(tqdm(data_gen)):
            prediction = model.compute(x)

            while isinstance(prediction, list):
                prediction = prediction[0]                

            submission.append({'idx': idx, 'label': prediction})
    else:
        raise ValueError(f'Unexpected SuperGLUE task name: {task_name}')

    save_path = output_path if output_path is not None else f'{SUPER_GLUE_TASKS[task_name]}.jsonl'
    save_path = expand_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(submission, save_path)
    log.info(f'Prediction saved to {save_path}')


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
