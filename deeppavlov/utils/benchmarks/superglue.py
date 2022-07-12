import argparse
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm import tqdm

from deeppavlov import build_model
from deeppavlov.core.commands.train import read_data_by_config, get_iterator_from_config
from deeppavlov.core.commands.utils import parse_config, expand_path
from deeppavlov.core.common.file import find_config, save_jsonl
from deeppavlov.download import deep_download

log = getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('config_path', help='path to a pipeline json config', type=str)
parser.add_argument('-o', '--output-file', default=None, help='path to save output', type=str)
parser.add_argument('-d', '--download', action='store_true', help='download model components')

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


def main():
    args = parser.parse_args()
    pipeline_config_path = find_config(args.config_path)
    if args.download:
        deep_download(pipeline_config_path)
    submit(pipeline_config_path, args.output_file)


if __name__ == '__main__':
    main()
