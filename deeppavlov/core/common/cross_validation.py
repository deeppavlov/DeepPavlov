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

from pathlib import Path
import sys
import numpy as np
import os

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.common.file import read_json, save_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.commands.train import train_evaluate_model_from_config, get_iterator_from_config, read_data_by_config
from sklearn.model_selection import KFold
from deeppavlov.core.common.params_search import ParamsSearch
from deeppavlov.core.commands.utils import expand_path

SAVE_PATH_ELEMENT_NAME = 'save_path'
TEMP_DIR_FOR_CV = 'cv_tmp'
log = get_logger(__name__)


def change_savepath_for_model(config):
    params_helper = ParamsSearch()

    dirs_for_saved_models=set()
    for p in params_helper.find_model_path(config, SAVE_PATH_ELEMENT_NAME):
        p.append(SAVE_PATH_ELEMENT_NAME)
        save_path = Path(params_helper.get_value_from_config(config, p))
        new_save_path = str(save_path.parent.joinpath(TEMP_DIR_FOR_CV).joinpath(save_path.name))

        dir = expand_path(os.path.dirname(new_save_path))
        dirs_for_saved_models.add(str(dir))
        os.makedirs(dir, exist_ok=True)

        config = params_helper.insert_value_or_dict_into_config(config, p, new_save_path)

    return config, dirs_for_saved_models

def delete_dir_for_saved_models(dirs_for_saved_models):
    for dir in dirs_for_saved_models:
        for root, dirs, files in os.walk(dir):
            for file in files:
                os.remove(os.path.join(root, file))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(dir)


def create_dirs_to_save_models(dirs_for_saved_models):
    for dir in dirs_for_saved_models:
        os.makedirs(dir, exist_ok=True)


def generate_train_valid(data, n_folds=5, is_loo=False):
    all_data = data['train'] + data['valid']

    # for Leave One Out
    if is_loo:
        for i in range(len(all_data)):
            data_i = {}
            data_i['train'] = all_data.copy()
            data_i['valid'] = [data_i['train'].pop(i)]
            data_i['test'] = []

            yield data_i
    # for Cross Validation
    else:
        kf = KFold(n_splits=n_folds, shuffle=True)
        for train_index, valid_index in kf.split(all_data):
            data_i = {}
            data_i['train'] = [all_data[i] for i in train_index]
            data_i['valid'] = [all_data[i] for i in valid_index]
            data_i['test'] = []

            yield data_i


def calc_cv_score(config=None, pipeline_config_path=None, data=None, n_folds=5, is_loo=False):
    if config is None:
        if pipeline_config_path is not None:
            config = read_json(pipeline_config_path)
        else:
            raise ValueError('Both \"config\" and \"pipeline_config_path\" is None')

    if data is None:
        data = read_data_by_config(config)

    config, dirs_for_saved_models = change_savepath_for_model(config)

    all_scores=[]
    target_metric = config['train']['metrics'][0]
    for data_i in generate_train_valid(data, n_folds=n_folds, is_loo=is_loo):
        iterator = get_iterator_from_config(config, data_i)
        create_dirs_to_save_models(dirs_for_saved_models)
        score = train_evaluate_model_from_config(config, iterator=iterator)
        delete_dir_for_saved_models(dirs_for_saved_models)
        all_scores.append(score['valid'][target_metric])

    cv_score = np.mean(all_scores)
    log.info('Cross-Validation \"{}\" is: {}'.format(target_metric, cv_score))

    return cv_score

