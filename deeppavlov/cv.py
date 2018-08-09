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

import argparse
from pathlib import Path
import sys
import numpy as np
from itertools import product

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.train import get_iterator_from_config
from deeppavlov.core.commands.train import read_data_by_config


log = get_logger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--config_path", help="path to a pipeline json config", type=str)
parser.add_argument("--loocv", help="path to a pipeline json config", type=bool, default=False)


def find_config(pipeline_config_path: str):
    if not Path(pipeline_config_path).is_file():
        configs = [c for c in Path(__file__).parent.glob(f'configs/**/{pipeline_config_path}.json')
                   if str(c.with_suffix('')).endswith(pipeline_config_path)]  # a simple way to not allow * and ?
        if configs:
            log.info(f"Interpreting '{pipeline_config_path}' as '{configs[0]}'")
            pipeline_config_path = str(configs[0])
    return pipeline_config_path


def calc_loocv_score(config, data):

    all_data = data['train'] + data['valid']
    m = len(all_data)
    all_scores = []
    for i in range(m):
        data_i = {}
        data_i['train'] = all_data.copy()
        data_i['valid'] = [data_i['train'].pop(i)]
        data_i['test'] = []
        iterator = get_iterator_from_config(config, data_i)
        score = train_evaluate_model_from_config(config, iterator=iterator)
        all_scores.append(score['valid']['accuracy'])

    return np.mean(all_scores)

def get_best_params(combinations, scores, param_names):
    max_id = np.argmax(scores)
    best_params = dict(zip(param_names, combinations[max_id]))
    best_params['score'] = scores[max_id]

    return best_params

def main():
    args = parser.parse_args()

    pipeline_config_path = find_config(args.config_path)
    config = read_json(pipeline_config_path)
    data = read_data_by_config(config)

    params = {}
    for elem in config['chainer']['pipe']:
        for key in elem.keys():
            if key.endswith('_range'):
                params[key.partition('_range')[0]] = elem[key]

    combinations = list(product(*params.values()))
    param_names = [k for k in params.keys()]

    scores=[]
    for comb in combinations:
        for i, param_value in enumerate(comb):
            for j, elem in enumerate(config['chainer']['pipe']):
                for key in elem.keys():
                    if key.partition('_range')[0] in param_names:
                        config['chainer']['pipe'][j][param_names[i]] = param_value

        if args.loocv:
            scores.append(calc_loocv_score(config, data))
        else:
            raise NotImplementedError('Not implemented this type of CV')

    print(get_best_params(combinations, scores, param_names))


if __name__ == "__main__":
    main()
