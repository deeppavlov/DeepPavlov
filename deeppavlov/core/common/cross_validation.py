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

import shutil
from collections import OrderedDict
from logging import getLogger
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from deeppavlov.core.commands.train import train_evaluate_model_from_config, get_iterator_from_config, \
    read_data_by_config
from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.params_search import ParamsSearch

SAVE_PATH_ELEMENT_NAME = 'save_path'
TEMP_DIR_FOR_CV = 'cv_tmp'
log = getLogger(__name__)


def change_savepath_for_model(config):
    params_helper = ParamsSearch()

    dirs_for_saved_models = set()
    for p in params_helper.find_model_path(config, SAVE_PATH_ELEMENT_NAME):
        p.append(SAVE_PATH_ELEMENT_NAME)
        save_path = Path(params_helper.get_value_from_config(config, p))
        new_save_path = save_path.parent / TEMP_DIR_FOR_CV / save_path.name

        dirs_for_saved_models.add(expand_path(new_save_path.parent))

        params_helper.insert_value_or_dict_into_config(config, p, str(new_save_path))

    return config, dirs_for_saved_models


def delete_dir_for_saved_models(dirs_for_saved_models):
    for new_save_dir in dirs_for_saved_models:
        shutil.rmtree(str(new_save_dir))


def create_dirs_to_save_models(dirs_for_saved_models):
    for new_save_dir in dirs_for_saved_models:
        new_save_dir.mkdir(exist_ok=True, parents=True)


def generate_train_valid(data, n_folds=5, is_loo=False):
    all_data = data['train'] + data['valid']

    if is_loo:
        # for Leave One Out
        for i in range(len(all_data)):
            data_i = {
                'train': all_data.copy(),
                'test': data['test']
            }
            data_i['valid'] = [data_i['train'].pop(i)]

            yield data_i
    else:
        # for Cross Validation
        kf = KFold(n_splits=n_folds, shuffle=True)
        for train_index, valid_index in kf.split(all_data):
            data_i = {
                'train': [all_data[i] for i in train_index],
                'valid': [all_data[i] for i in valid_index],
                'test': data['test']
            }

            yield data_i


def calc_cv_score(config, data=None, n_folds=5, is_loo=False):
    config = parse_config(config)

    if data is None:
        data = read_data_by_config(config)

    config, dirs_for_saved_models = change_savepath_for_model(config)

    cv_score = OrderedDict()
    for data_i in generate_train_valid(data, n_folds=n_folds, is_loo=is_loo):
        iterator = get_iterator_from_config(config, data_i)
        create_dirs_to_save_models(dirs_for_saved_models)
        score = train_evaluate_model_from_config(config, iterator=iterator)
        delete_dir_for_saved_models(dirs_for_saved_models)
        for key, value in score['valid'].items():
            if key not in cv_score:
                cv_score[key] = []
            cv_score[key].append(value)

    for key, value in cv_score.items():
        cv_score[key] = np.mean(value)
        log.info('Cross-Validation \"{}\" is: {}'.format(key, cv_score[key]))

    return cv_score
