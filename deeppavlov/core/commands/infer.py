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

import pickle
import sys

import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from logging import getLogger
from typing import Iterable, Optional, Union, Any
from collections import defaultdict

from deeppavlov.core.commands.utils import import_packages, parse_config
from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.file import save_jsonl
from deeppavlov.core.commands.train import read_data_by_config, get_iterator_from_config
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
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
                      file_path: Optional[str] = None,
                      data_type: Optional[str] = 'test',
                      iterator: Union[DataLearningIterator, DataFittingIterator] = None) -> None:
    """Make a prediction with the component described in corresponding configuration file."""
    config = parse_config(config)

    if iterator is None:
        data = read_data_by_config(config)
        iterator = get_iterator_from_config(config, data)

    data_gen = iterator.gen_batches(1, data_type=data_type, shuffle=False)
    
    try:
        task_name = config['dataset_reader']['name']
    except KeyError:
        task_name = 'test'

    model = build_model(config)

    submission = []

    if task_name == 'record' or task_name == "rucos":
        index_list = []
        entities_list = []
        predictions = []

        for idx, (x, _) in enumerate(tqdm(data_gen)):
            indices = x[0][0]
            queries = x[0][1]
            passages = x[0][2]
            entities = x[0][3]
            num_examples = x[0][4]

            prediction = model.compute(x)[:, 1]                

            index_list.append(indices)
            entities_list.append(entities)
            predictions.append(prediction)

        output = defaultdict(
            lambda: {
                "predicted": [],
                "probability": []
            }
        )
        
        for index, entity, prediction in zip(index_list, entities_list, predictions):
            output[index]["predicted"].append(entity)
            output[index]["probability"].append(prediction)
        
        for key, value in output.items():
            answer_index = np.argmax(value["probability"])
            answer = value["predicted"][answer_index]
            submission.append(
                {
                    "idx": int(key.split("-")[-2]),
                    "label": answer
                }
            )

    elif task_name == 'copa' or task_name == 'parus':
        for idx, (x, _) in enumerate(tqdm(data_gen)):
            prediction = model.compute(x)
            
            if prediction[0] == "choice1":
                label = 0
            else:
                label = 1

            submission.append({'idx': idx, 'label': label})

    elif task_name == 'muserc' or task_name == 'multirc':
        tmp_output = []
        for idx, (x, _) in enumerate(tqdm(data_gen)):
            contexts = x[0][0]
            answers = x[0][1]

            indices = x[0][2]
            paragraph_ind = indices['paragraph']
            question_ind = indices['question']
            answer_ind = indices['answer']

            prediction = model(contexts, answers)

            tmp_output.append(dict(paragraph=int(paragraph_ind), 
                                   question=int(question_ind), 
                                   answer=int(answer_ind), 
                                   label=prediction[0]))

        output = {}
        for line in tmp_output:
            paragraph_idx = line["paragraph"]
            question_idx = line["question"]
            answer_idx = line["answer"]
            label = 1 if line["label"] == "True" else 0
            if paragraph_idx not in output:
                output[paragraph_idx] = dict(
                    idx=paragraph_idx, passage=dict(
                        questions=[
                            dict(
                                idx=question_idx,
                                answers=[
                                    dict(
                                        idx=answer_idx,
                                        label=label
                                    )
                                ]
                            )
                        ]
                    )
                )
    
            questions = output[paragraph_idx]["passage"]["questions"]
            question_indices = set(el["idx"] for el in questions)
            if question_idx not in question_indices:
                output[paragraph_idx]["passage"]["questions"].append(dict(
                    idx=question_idx, answers=[dict(idx=answer_idx, label=label)]
                ))
            else:
                for question in output[paragraph_idx]["passage"]["questions"]:
                    if question["idx"] == question_idx:
                        question["answers"].append(dict(idx=answer_idx, label=label))
    
        submission = [value for _, value in output.items()]

    else:
        for idx, (x, _) in enumerate(tqdm(data_gen)):
            prediction = model.compute(x)

            while isinstance(prediction, list):
                prediction = prediction[0]                

            submission.append({'idx': idx, 'label': prediction})

    save_jsonl(submission, task_name)
