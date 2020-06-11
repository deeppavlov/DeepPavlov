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
import sys
from copy import deepcopy
from itertools import product
from logging import getLogger
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from deeppavlov.core.commands.train import train_evaluate_model_from_config, get_iterator_from_config, \
    read_data_by_config
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.core.common.cross_validation import calc_cv_score
from deeppavlov.core.common.file import save_json, find_config, read_json
from deeppavlov.core.common.params_search import ParamsSearch

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

log = getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="path to a pipeline json config", type=str)
parser.add_argument("--folds", help="number of folds", type=str, default=None)
parser.add_argument("--search_type", help="search type: grid or random search", type=str, default='grid')


def get_best_params(combinations, scores, param_names, target_metric):
    max_id = np.argmax(scores)
    best_params = dict(zip(param_names, combinations[max_id]))
    best_params[target_metric] = scores[max_id]

    return best_params


def main():
    params_helper = ParamsSearch()

    args = parser.parse_args()
    is_loo = False
    n_folds = None
    if args.folds == 'loo':
        is_loo = True
    elif args.folds is None:
        n_folds = None
    elif args.folds.isdigit():
        n_folds = int(args.folds)
    else:
        raise NotImplementedError('Not implemented this type of CV')

    # read config
    pipeline_config_path = find_config(args.config_path)
    config_init = read_json(pipeline_config_path)
    config = parse_config(config_init)
    data = read_data_by_config(config)
    target_metric = parse_config(config_init)['train']['metrics'][0]
    if isinstance(target_metric, dict):
        target_metric = target_metric['name']

    # get all params for search
    param_paths = list(params_helper.find_model_path(config, 'search_choice'))
    param_values = []
    param_names = []
    for path in param_paths:
        value = params_helper.get_value_from_config(config, path)
        param_name = path[-1]
        param_value_search = value['search_choice']
        param_names.append(param_name)
        param_values.append(param_value_search)

    # find optimal params
    if args.search_type == 'grid':
        # generate params combnations for grid search
        combinations = list(product(*param_values))

        # calculate cv scores
        scores = []
        for comb in combinations:
            config = deepcopy(config_init)
            for param_path, param_value in zip(param_paths, comb):
                params_helper.insert_value_or_dict_into_config(config, param_path, param_value)
            config = parse_config(config)

            if (n_folds is not None) | is_loo:
                # CV for model evaluation
                score_dict = calc_cv_score(config, data=data, n_folds=n_folds, is_loo=is_loo)
                score = score_dict[next(iter(score_dict))]
            else:
                # train/valid for model evaluation
                data_to_evaluate = data.copy()
                if len(data_to_evaluate['valid']) == 0:
                    data_to_evaluate['train'], data_to_evaluate['valid'] = train_test_split(data_to_evaluate['train'],
                                                                                            test_size=0.2)
                iterator = get_iterator_from_config(config, data_to_evaluate)
                score = train_evaluate_model_from_config(config, iterator=iterator)['valid'][target_metric]

            scores.append(score)

        # get model with best score
        best_params_dict = get_best_params(combinations, scores, param_names, target_metric)
        log.info('Best model params: {}'.format(best_params_dict))
    else:
        raise NotImplementedError('Not implemented this type of search')

    # save config
    best_config = config_init
    for i, param_name in enumerate(best_params_dict.keys()):
        if param_name != target_metric:
            params_helper.insert_value_or_dict_into_config(best_config, param_paths[i], best_params_dict[param_name])

    best_model_filename = pipeline_config_path.with_suffix('.cvbest.json')
    save_json(best_config, best_model_filename)
    log.info('Best model saved in json-file: {}'.format(best_model_filename))


# try to run:
# --config_path path_to_config.json --folds 2
if __name__ == "__main__":
    main()
