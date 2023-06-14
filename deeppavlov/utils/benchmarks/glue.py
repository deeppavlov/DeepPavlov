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
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tqdm import tqdm

from deeppavlov import build_model
from deeppavlov.core.commands.train import read_data_by_config, get_iterator_from_config
from deeppavlov.core.commands.utils import parse_config, expand_path
from deeppavlov.core.common.file import find_config
from deeppavlov.download import deep_download

log = getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('config_path', help='path to a pipeline json config', type=str)
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


def get_predictions(model, data_gen, replace_word=None, round_res=False):
    submission = {'index': [], 'prediction': []}
    for idx, (x, _) in enumerate(tqdm(data_gen)):
        prediction = model.compute(x)[0]
        if replace_word:
            prediction = 1 if prediction in {replace_word} else 0
        if round_res:
            prediction = round(prediction, 3)
        submission['index'].append(idx)
        submission['prediction'].append(prediction)
    return submission


def submit(config: Union[str, Path, dict], output_path: Optional[Union[str, Path]] = None) -> None:
    """Creates submission file for the GLUE tasks.
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

    if task_name in {'cola'}:
        submission = get_predictions(model, data_gen, 'acceptable')

    elif task_name in {'mnli'}:
        if config['dataset_reader']['valid'] in {'validation_matched'}:
            task_name = 'mnli-m'
        else:
            task_name = 'mnli-mm'
        submission = get_predictions(model, data_gen)

    elif task_name in {'mrpc'}:
        submission = get_predictions(model, data_gen, 'equivalent')

    elif task_name in {'sst2'}:
        submission = get_predictions(model, data_gen, 'positive')

    elif task_name in {'stsb'}:
        submission = get_predictions(model, data_gen, None, True)

    elif task_name in {'wnli'}:
        submission = get_predictions(model, data_gen, 'entailment')

    elif task_name in GLUE_TASKS:
        submission = get_predictions(model, data_gen)
    else:
        raise ValueError(f'Unexpected GLUE task name: {task_name}')

    save_path = output_path if output_path is not None else f'{GLUE_TASKS[task_name]}.tsv'
    save_path = expand_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(submission)
    df.to_csv(save_path, sep="\t", index=False)
    log.info(f'Prediction saved to {save_path}')


def main():
    args = parser.parse_args()
    pipeline_config_path = find_config(args.config_path)
    if args.download:
        deep_download(pipeline_config_path)
    submit(pipeline_config_path, args.output_file)


if __name__ == '__main__':
    main()
