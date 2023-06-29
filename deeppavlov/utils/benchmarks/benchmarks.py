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

import argparse
from collections import defaultdict
from logging import getLogger

import numpy as np
from tqdm import tqdm

from deeppavlov import build_model
from deeppavlov.core.commands.train import read_data_by_config, get_iterator_from_config
from deeppavlov.core.commands.utils import parse_config, expand_path
from deeppavlov.core.common.file import save_jsonl

log = getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('config_path', help='path to a pipeline json config', type=str)
parser.add_argument('benchmark_name', help='benchmark name to be submitted',
                    choices=['glue', 'superglue', 'russian_superglue'])
parser.add_argument('-o', '--output-file', default=None, help='path to save output', type=str)
parser.add_argument('-d', '--download', action='store_true', help='download model components')

GLUE_TASKS = {
    'cola': 'CoLA',
    'mnli-m': 'MNLI-m',
    'mnli-mm': 'MNLI-mm',
    'mrpc': 'MRPC',
    'qnli': 'QNLI',
    'qqp': 'QQP',
    'rte': 'RTE',
    'sst2': 'SST-2',
    'stsb': 'STS-B',
    'wnli': 'WNLI'
}

SUPER_GLUE_TASKS = {
    'copa': 'COPA',
    'multirc': 'MultiRC',
    'boolq': 'BoolQ',
    'record': 'ReCoRD',
    'wic': 'WiC'
}

RSG_TASKS = {
    'lidirus': 'LiDiRus',
    'rcb': 'RCB',
    'parus': 'PARus',
    'muserc': 'MuSeRC',
    'terra': 'TERRa',
    'russe': 'RUSSE',
    'rwsd': 'RWSD',
    'danetqa': 'DaNetQA',
    'rucos': 'RuCoS'
}


def split_config(config_path, download):
    """Gets model, data iterator and a task name from the configuration file.
    
    Args:
        config_path: Path to the model configuration file.
        download: If True, the model will be downloaded from the DeepPavlov server.
    """

    config = parse_config(config_path)
    data = read_data_by_config(config)
    iterator = get_iterator_from_config(config, data)
    task_name = config['dataset_reader']['name']
    if task_name == 'mnli':
        task_name = 'mnli-m' if config['dataset_reader']['valid'] == 'validation_matched' else 'mnli-mm'
    data_gen = iterator.gen_batches(1, data_type='test', shuffle=False)
    model = build_model(config, download=download)
    return model, data_gen, task_name


def get_predictions(model, data_gen, replace_word=None, round_res=False):
    """Gets model predictions and replaces model output with replace_word.
    
    Args:
        model: The model itself.
        data_gen: Iterator with data to be submitted.
        replace_word: Model outputs to be replaced with 1, other outputs are replaced with 0.
            If None, model outputs are not replaced.
        round_res: If True, model outputs are rounded (used in stsb).
    """

    submission = {'index': [], 'prediction': []}
    for idx, (x, _) in enumerate(tqdm(data_gen)):
        prediction = model.compute(x)[0]
        if replace_word:
            prediction = 1 if prediction == replace_word else 0
        if round_res:
            prediction = round(prediction, 3)
        submission['index'].append(idx)
        submission['prediction'].append(prediction)
    return submission


def submit_glue(config_path, output_path, download):
    """Creates submission file for the GLUE tasks.
    Args:
        config_path: Path to the model configuration file.
        output_path: Path to output file. If None, file name is selected according corresponding task name.
        download: If True, the model will be downloaded from the DeepPavlov server.
    """

    model, data_gen, task_name = split_config(config_path, download)

    if task_name == 'cola':
        submission = get_predictions(model, data_gen, 'acceptable')

    elif task_name.startswith('mnli'):
        submission = get_predictions(model, data_gen)

    elif task_name == 'mrpc':
        submission = get_predictions(model, data_gen, 'equivalent')

    elif task_name == 'sst2':
        submission = get_predictions(model, data_gen, 'positive')

    elif task_name == 'stsb':
        submission = get_predictions(model, data_gen, None, True)

    elif task_name == 'wnli':
        submission = get_predictions(model, data_gen, 'entailment')

    elif task_name in GLUE_TASKS:
        submission = get_predictions(model, data_gen)
    else:
        raise ValueError(f'Unexpected GLUE task name: {task_name}')

    save_path = output_path or f'{GLUE_TASKS[task_name]}.tsv'
    save_path = expand_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_array = np.vstack(([list(submission.keys())], np.array(list(submission.values())).transpose()))
    np.savetxt(save_path, save_array, delimiter='\t', fmt='%s')
    log.info(f'Prediction saved to {save_path}')


def commonsense_reasoning_prediction(model, data_gen):
    """Common part for ReCoRD and RuCoS tasks that gets their predictions in needed format.
    
    Args:
        model: The model itself.
        data_gen: Iterator with data to be submitted.
    """

    submission = []
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
    return submission


def multi_sentence_comprehension_prediction(model, data_gen):
    """Common part for MultiRC and MuSeRC tasks that gets their predictions in needed format.
    
    Args:
        model: The model itself.
        data_gen: Iterator with data to be submitted.
    """

    output = {}

    for x, _ in tqdm(data_gen):
        contexts, answers, indices = x[0]

        prediction = model([contexts], [answers], indices)

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
    return submission


def submit_superglue(config_path, output_path, download):
    """Creates submission file for the SuperGLUE tasks.

    Args:
        config_path: Path to the model configuration file.
        output_path: Path to output file. If None, file name is selected according corresponding task name.
        download: If True, the model will be downloaded from the DeepPavlov server.
    """

    model, data_gen, task_name = split_config(config_path, download)
    submission = []

    if task_name == 'record':
        submission = commonsense_reasoning_prediction(model, data_gen)

    elif task_name == 'copa':
        for idx, (x, _) in enumerate(tqdm(data_gen)):
            prediction = model.compute(x)[0]
            label = int(prediction == 'choice2')
            submission.append({'idx': idx, 'label': label})

    elif task_name == 'multirc':
        submission = multi_sentence_comprehension_prediction(model, data_gen)

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


def submit_rsg(config_path, output_path, download):
    """Creates submission file for the Russian SuperGLUE tasks.

    Args:
        config_path: Path to the model configuration file.
        output_path: Path to output file. If None, file name is selected according corresponding task name.
        download: If True, the model will be downloaded from the DeepPavlov server.
    """

    model, data_gen, task_name = split_config(config_path, download)
    submission = []

    if task_name == 'rucos':
        submission = commonsense_reasoning_prediction(model, data_gen)

    elif task_name == 'parus':
        for idx, (x, _) in enumerate(tqdm(data_gen)):
            prediction = model.compute(x)[0]
            label = int(prediction == 'choice2')
            submission.append({'idx': idx, 'label': label})

    elif task_name == 'muserc':
        submission = multi_sentence_comprehension_prediction(model, data_gen)

    elif task_name in RSG_TASKS:
        for idx, (x, _) in enumerate(tqdm(data_gen)):
            prediction = model.compute(x)

            while isinstance(prediction, list):
                prediction = prediction[0]

            submission.append({'idx': idx, 'label': prediction})
    else:
        raise ValueError(f'Unexpected Russian SuperGLUE task name: {task_name}')

    save_path = output_path if output_path is not None else f'{RSG_TASKS[task_name]}.jsonl'
    save_path = expand_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(submission, save_path)
    log.info(f'Prediction saved to {save_path}')


def main():
    args = parser.parse_args()
    if args.benchmark_name == 'glue':
        submit_glue(args.config_path, args.output_file, args.download)
    elif args.benchmark_name == 'superglue':
        submit_superglue(args.config_path, args.output_file, args.download)
    elif args.benchmark_name == 'russian_superglue':
        submit_rsg(args.config_path, args.output_file, args.download)


if __name__ == '__main__':
    main()
