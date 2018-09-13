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

from time import time
from datetime import datetime
from os.path import join
from copy import copy
from shutil import rmtree
from typing import Union, Dict

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.cross_validation import calc_cv_score
from deeppavlov.pipeline_manager.pipegen import PipeGen
from deeppavlov.pipeline_manager.utils import normal_time
from deeppavlov.pipeline_manager.logger import Logger
from deeppavlov.pipeline_manager.utils import results_visualization


class PipelineManager:
    """
    The class implements the functions of automatic pipeline search and search for hyperparameters.

    Args:
            config_path: path to config file.
            exp_name: name of the experiment.
            date: date of the experiment.
            mode: train or evaluate - the trigger that determines the operation of the algorithm
            info: some additional information that you want to add to the log, the content of the dictionary
                  does not affect the algorithm
            root: root path, the root path where the report will be generated and saved checkpoints
            hyper_search: grid or random - the trigger that determines type of hypersearch
            sample_num: determines the number of generated pipelines, if hyper_search == random.
            target_metric: The metric name on the basis of which the results will be sorted when the report
                           is generated. The default value is None, in this case the target metric is taken the
                           first name from those names that are specified in the config file. If the specified metric
                           is not contained in DeepPavlov will be called error.
            plot: boolean trigger, which determines whether to draw a graph of results or not

    Attributes:
        logger: A special class that collects auxiliary statistics and results during training, and stores all
                the collected data in a separate log.
        pipeline_generator: A special class that generates configs for training.
    """
    def __init__(self,
                 config_path: str,
                 exp_name: str,
                 date: Union[str, None] = None,
                 mode: str = 'train',
                 info: Dict = None,
                 root: str = './experiments/',
                 cross_val: bool = False,
                 k_fold: Union[int, None] = 5,
                 search: bool = False,
                 hyper_search: str = 'random',
                 sample_num: int = 10,
                 target_metric: str = None,
                 plot: bool = True):
        """
        Initialize logger, read input args, builds a directory tree, initialize date.
        """
        self.config_path = config_path
        self.exp_name = exp_name
        self.mode = mode
        self.info = info
        self.cross_validation = cross_val
        self.k_fold = k_fold
        self.search = search
        self.hyper_search = hyper_search
        self.sample_num = sample_num
        self.target_metric = target_metric
        self.plot = plot
        self.pipeline_generator = None
        if date is not None:
            self.date = date
        else:
            date_ = datetime.now()
            self.date = '{}-{}-{}'.format(date_.year, date_.month, date_.day)

        self.root = root
        self.save_path = join(self.root, self.date, self.exp_name, 'checkpoints')

        self.logger = Logger(exp_name, root, self.info, self.date, self.plot)
        self.start_exp = time()
        # start test
        self.test()

    def run(self):
        """
        Initializes the pipeline generator and runs the experiment. Creates a report after the experiments.
        """
        # create the pipeline generator
        self.pipeline_generator = PipeGen(self.config_path, self.save_path, n=self.sample_num, search=self.search,
                                          search_type=self.hyper_search, test_mode=False,
                                          cross_val=self.cross_validation)

        # Start generating pipelines configs
        print('[ Experiment start - {0} pipes, will be run]'.format(self.pipeline_generator.length))
        if self.cross_validation:
            print("[ WARNING: Cross validation is active! Every pipeline will be run {0} times! ]".format(self.k_fold))

        self.logger.log['experiment_info']['number_of_pipes'] = self.pipeline_generator.length
        exp_start_time = time()
        for i, pipe in enumerate(self.pipeline_generator()):
            # print progress
            if i != 0:
                itime = normal_time(((time() - exp_start_time) / i) * (self.pipeline_generator.length - i))
                ptime = normal_time(time() - exp_start_time)
                print('\n')
                print('[ Progress: pipe {0}/{1}; Time pass: {2} ;'
                      ' Time left: {3}; ]'.format(i+1, self.pipeline_generator.length, ptime, itime))

                self.logger.log['experiment_info']['metrics'] = copy(pipe['train']['metrics'])
                self.logger.log['experiment_info']['target_metric'] = self.target_metric

            self.logger.pipe_ind = i + 1
            self.logger.pipe_conf = copy(pipe['chainer']['pipe'])

            # start pipeline time
            pipe_start = time()
            if self.cross_validation:
                cv_score = calc_cv_score(pipe, n_folds=self.k_fold)
                results = {"test": cv_score}
            else:
                if self.mode == 'train':
                    results = train_evaluate_model_from_config(pipe, to_train=True, to_validate=True)

                elif self.mode == 'evaluate':
                    results = train_evaluate_model_from_config(pipe, to_train=False, to_validate=False)
                else:
                    raise ValueError("Only 'train' and 'evaluate' mode are available,"
                                     " but {0} was found.".format(self.mode))

            # update logger
            self.logger.pipe_time = normal_time(time() - pipe_start)
            self.logger.pipe_res = results
            self.logger.get_pipe_log()

            # save config in checkpoint folder
            if not self.cross_validation:
                self.save_config(pipe, i)

        # save log
        self.logger.log['experiment_info']['full_time'] = normal_time(time() - self.start_exp)
        self.logger.save()

        # visualization of results
        path = join(self.root, self.date, self.exp_name)
        results_visualization(path, join(path, 'images'), self.plot, self.target_metric)
        return None

    def test(self):
        """
        Initializes the pipeline generator with tiny data and runs the test of experiment.
        """
        # create the pipeline generator
        pipeline_generator = PipeGen(self.config_path, self.save_path, n=self.sample_num, search=self.search,
                                     search_type=self.hyper_search, test_mode=True)

        # Start generating pipelines configs
        print('[ Test start - {0} pipes, will be run]'.format(pipeline_generator.length))
        exp_start_time = time()
        for i, pipe in enumerate(pipeline_generator()):
            # print progress
            if i != 0:
                itime = normal_time(((time() - exp_start_time) / i) * (pipeline_generator.length - i))
                print('\n')
                print('[ Test progress: pipe {0}/{1}; Time left: {2}; ]'.format(i + 1,
                                                                                pipeline_generator.length, itime))

            if pipe['dataset_reader']['name'] == 'basic_classification_reader':
                pipe['dataset_reader'] = {
                    "name": "basic_classification_reader",
                    "x": "text",
                    "y": "target",
                    "data_path": '../tests/test_data/classification_data/'
                }
            else:
                raise ConfigError("Dataset reader is not intended for classification task."
                                  "Name of dataset_reader must be 'basic_classification_reader',"
                                  "but {} was found in config.".format(pipe['dataset_reader']['name']))

            if self.mode == 'train':
                results = train_evaluate_model_from_config(pipe, to_train=True, to_validate=False)
            elif self.mode == 'evaluate':
                results = train_evaluate_model_from_config(pipe, to_train=False, to_validate=False)
            else:
                raise ValueError("Only 'train' and 'evaluate' mode are available, but {0} was found.".format(self.mode))

            del results

        # del all tmp files in save path
        rmtree(join(self.save_path, "tmp"))
        print('[ The test was successful ]')
        return None

    def save_config(self, conf, i) -> None:
        """
        Save train config in checkpoint folder.
        """
        with open(join(self.save_path, "pipe_{}".format(i+1), 'config.json'), 'w') as cf:
            json.dump(conf, cf)
            cf.close()
