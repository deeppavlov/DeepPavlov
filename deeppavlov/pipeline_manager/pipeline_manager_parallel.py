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

import os
import json
import shutil

from time import time
from tqdm import tqdm
from os.path import join
from shutil import rmtree
from psutil import cpu_count
from datetime import datetime
from copy import copy, deepcopy
from multiprocessing import Pool
from typing import Union, Dict, List

from deeppavlov.core.common.file import read_json
from deeppavlov.pipeline_manager.logger import Logger
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.pipeline_manager.pipegen import PipeGen
from deeppavlov.pipeline_manager.utils import normal_time
from deeppavlov.pipeline_manager.utils import get_num_gpu
from deeppavlov.core.common.prints import RedirectedPrints
from deeppavlov.core.common.cross_validation import calc_cv_score
from deeppavlov.core.data.data_fitting_iterator import DataFittingIterator
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.train import read_data_by_config, get_iterator_from_config
from deeppavlov.pipeline_manager.utils import results_visualization, get_available_gpus


def unpack_args(func):
    from functools import wraps

    @wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        else:
            return func(*args)
    return wrapper


class PipelineManager:
    """
    The class implements the functions of automatic pipeline search and search for hyperparameters.

    Args:
        config_path: path to config file.
        exp_name: name of the experiment.

    Attributes:
        date: date of the experiment.
        info: some additional information that you want to add to the log, the content of the dictionary
              does not affect the algorithm
        root: root path, the root path where the report will be generated and saved checkpoints
        sample_num: determines the number of generated pipelines, if hyper_search == random.
        target_metric: The metric name on the basis of which the results will be sorted when the report
                       is generated. The default value is None, in this case the target metric is taken the
                       first name from those names that are specified in the config file. If the specified metric
                       is not contained in DeepPavlov will be called error.
        logger: A special class that collects auxiliary statistics and results during training, and stores all
                the collected data in a separate log.
        plot: boolean trigger, which determines whether to draw a graph of results or not
        pipeline_generator: A special class that generates configs for training.
    """
    def __init__(self, config_path: Union[str, Dict], exp_name: str):
        """
        Initialize logger, read input args, builds a directory tree, initialize date.
        """
        if isinstance(config_path, str):
            self.exp_config = read_json(config_path)
        else:
            self.exp_config = config_path

        self.exp_name = exp_name
        self.date = self.exp_config['enumerate'].get('date', datetime.now().strftime('%Y-%m-%d'))
        self.info = self.exp_config['enumerate'].get('info')
        self.root = self.exp_config['enumerate'].get('root', 'experiments/')
        self.plot = self.exp_config['enumerate'].get('plot', False)
        self.save_best = self.exp_config['enumerate'].get('save_best', False)
        self.do_test = self.exp_config['enumerate'].get('do_test', False)
        self.cross_validation = self.exp_config['enumerate'].get('cross_val', False)
        self.k_fold = self.exp_config['enumerate'].get('k_fold', 5)
        self.sample_num = self.exp_config['enumerate'].get('sample_num', 10)
        self.target_metric = self.exp_config['enumerate'].get('target_metric')
        self.multiprocessing = self.exp_config['enumerate'].get('multiprocessing', True)
        self.max_num_workers_ = self.exp_config['enumerate'].get('max_num_workers')
        self.use_all_gpus = self.exp_config['enumerate'].get('use_all_gpus', False)
        self.use_multi_gpus = self.exp_config['enumerate'].get('use_multi_gpus')
        self.memory_fraction = self.exp_config['enumerate'].get('gpu_memory_fraction', 1.0)

        self.pipeline_generator = None
        self.gen_len = 0

        # Logger initialization
        self.save_path = join(self.root, self.date, self.exp_name, 'checkpoints')
        self.logger = Logger(self.exp_name, self.root, self.info, self.date, self.plot)

        # multiprocessing
        if self.use_multi_gpus and self.use_all_gpus:
            raise ValueError("Parameters 'use_all_gpus' and 'use_multi_gpus' can not simultaneously be not None.")

        self.max_num_workers = None
        self.available_gpu = None
        if self.multiprocessing:
            self.prepare_multiprocess()

        # write time of experiment start
        self.start_exp = time()
        # start test
        if self.do_test:
            self.dataset_composition = dict(train=False, valid=False, test=False)
            self.test()

    def prepare_multiprocess(self):
        cpu_num = cpu_count()
        gpu_num = get_num_gpu()

        try:
            visible_gpu = [int(q) for q in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        except KeyError:
            visible_gpu = []

        if self.max_num_workers_ is not None and self.max_num_workers_ < 1:
            raise ConfigError("The number of workers must be at least equal to one. "
                              "Please check 'max_num_workers' parameter in config.")

        if self.use_all_gpus:
            if self.max_num_workers_ is None:
                self.available_gpu = get_available_gpus(gpu_fraction=self.memory_fraction)
                self.available_gpu = list(set(self.available_gpu) & set(visible_gpu))

                if len(self.available_gpu) == 0:
                    raise ValueError("GPU with numbers: ({}) are busy.".format(set(visible_gpu)))
                elif len(self.available_gpu) < len(visible_gpu):
                    print("PipelineManagerWarning: 'CUDA_VISIBLE_DEVICES' = ({0}), "
                          "but only {1} are available.".format(visible_gpu, self.available_gpu))

                if int(cpu_num * 0.7) > len(self.available_gpu):
                    self.max_num_workers = len(self.available_gpu)
                else:
                    self.max_num_workers = int(cpu_num * 0.7)
            else:
                if self.max_num_workers_ > gpu_num:
                    self.max_num_workers = gpu_num
                    self.available_gpu = get_available_gpus(gpu_fraction=self.memory_fraction)
                    self.available_gpu = list(set(self.available_gpu) & set(visible_gpu))

                    if len(self.available_gpu) == 0:
                        raise ValueError("GPU with numbers: ({}) are busy.".format(set(visible_gpu)))
                else:
                    self.available_gpu = get_available_gpus(num_gpus=self.max_num_workers_,
                                                            gpu_fraction=self.memory_fraction)
                    self.available_gpu = list(set(self.available_gpu) & set(visible_gpu))

                    if len(self.available_gpu) == 0:
                        raise ValueError("GPU with numbers: ({}) are busy.".format(set(visible_gpu)))

                    self.max_num_workers = len(self.available_gpu)

        elif self.use_multi_gpus:
            self.use_multi_gpus = list(set(self.use_multi_gpus) & set(visible_gpu))
            if len(self.use_multi_gpus) == 0:
                raise ValueError("GPU numbers in 'use_multi_gpus' and 'CUDA_VISIBLE_DEVICES' "
                                 "has not intersections".format(set(visible_gpu)))

            self.available_gpu = get_available_gpus(gpu_select=self.use_multi_gpus, gpu_fraction=self.memory_fraction)

            if len(self.available_gpu) == 0:
                raise ValueError("All GPU from 'use_multi_gpus' are busy. "
                                 "GPU numbers ({});".format(self.use_multi_gpus))

            if not self.max_num_workers_:
                self.max_num_workers = len(self.available_gpu)
            else:
                if self.max_num_workers_ > len(self.available_gpu):
                    self.max_num_workers = len(self.available_gpu)
                else:
                    self.max_num_workers = self.max_num_workers_
                    self.available_gpu = self.available_gpu[0:self.max_num_workers_]

        else:
            if self.max_num_workers_ is not None and self.max_num_workers_ > cpu_num:
                print("PipelineManagerWarning: parameter 'max_num_workers'={0}, "
                      "but amounts of cpu is {1}. The {1} will be assigned to 'max_num_workers' "
                      "as default.".format(self.max_num_workers_, cpu_num))
                self.max_num_workers_ = cpu_num

            self.max_num_workers = self.max_num_workers_

    @staticmethod
    @unpack_args
    def train_pipe(pipe_config, cross_validation, k_fold, gpu, gpu_ind, pipe_ind, out_path):
        # modify project environment
        if gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        # start pipeline
        pipe_start = time()
        res = dict(pipe_conf=deepcopy(pipe_config))

        #
        dataset_name = copy(res['pipe_conf']['dataset_reader']['data_path'])
        save_path = join(out_path, dataset_name, "pipe_{}".format(pipe_ind + 1))
        os.makedirs(save_path)
        #
        with RedirectedPrints(out=open(join(save_path, "out.txt"), "w")):
            if cross_validation:
                cv_score = calc_cv_score(pipe_config, n_folds=k_fold)
                results = {"test": cv_score}
            else:
                results = train_evaluate_model_from_config(pipe_config, to_train=True, to_validate=True)

        res['results'] = results
        res['pipe_time'] = time() - pipe_start
        return res

    def update_logger(self, res_list: List[dict]):
        dataset_res = {}
        for i, res in enumerate(res_list):
            pipe = res['pipe_conf']
            results = res['results']
            dataset_name = copy(pipe['dataset_reader']['data_path'])

            if i == 0:
                self.logger.log['experiment_info']['metrics'] = copy(pipe['train']['metrics'])
                if self.target_metric is None:
                    self.target_metric = pipe['train']['metrics'][0]['name']
                self.logger.log['experiment_info']['target_metric'] = self.target_metric

            self.logger.pipe_ind = i + 1
            self.logger.pipe_conf = copy(pipe['chainer']['pipe'])
            self.logger.dataset = dataset_name
            self.logger.batch_size = pipe['train'].get('batch_size', "None")

            # add results and pipe time to log
            self.logger.pipe_time = normal_time(res['pipe_time'])
            self.logger.pipe_res = results
            # update logger
            self.logger.get_pipe_log()
            self.logger.write()
            # save config in checkpoint folder
            if not self.cross_validation:
                self.save_config(pipe, dataset_name, i)

            if self.save_best:
                if dataset_name not in dataset_res.keys():
                    dataset_res[dataset_name] = dict(best_score=-1, best_ind=None)

                if 'test' in results.keys():
                    if results['test'][self.target_metric] > dataset_res[dataset_name]["best_score"]:
                        dataset_res[dataset_name]["best_score"] = results['test'][self.target_metric]
                        dataset_res[dataset_name]["best_ind"] = i + 1

                else:
                    if results['valid'][self.target_metric] > dataset_res[dataset_name]["best_score"]:
                        dataset_res[dataset_name]["best_score"] = results['valid'][self.target_metric]
                        dataset_res[dataset_name]["best_ind"] = i + 1

        return dataset_res

    def gpu_gen(self, gpu=False):
        if gpu:
            for i, pipe_conf in enumerate(self.pipeline_generator()):
                j = i - (i // len(self.available_gpu)) * len(self.available_gpu)
                yield (deepcopy(pipe_conf), self.cross_validation, self.k_fold, True, j, i, self.save_path)
        else:
            for i, pipe_conf in enumerate(self.pipeline_generator()):
                yield (deepcopy(pipe_conf), self.cross_validation, self.k_fold, False, None, i, self.save_path)

    def run(self):
        """
        Initializes the pipeline generator and runs the experiment. Creates a report after the experiments.
        """
        # create the pipeline generator
        self.pipeline_generator = PipeGen(self.exp_config, self.save_path, sample_num=self.sample_num, test_mode=False,
                                          cross_val=self.cross_validation)
        self.gen_len = self.pipeline_generator.length

        # Start generating pipelines configs
        print('[ Experiment start - {0} pipes, will be run]'.format(self.gen_len))
        if self.cross_validation:
            print("[ WARNING: Cross validation is active! Every pipeline will be run {0} times! ]".format(self.k_fold))

        self.logger.log['experiment_info']['number_of_pipes'] = self.gen_len

        if self.multiprocessing:
            # start multiprocessing
            workers = Pool(self.max_num_workers)

            if self.available_gpu is None:
                pipes_results = workers.imap_unordered(self.train_pipe, [x for x in self.gpu_gen(gpu=False)])
                workers.close()
                workers.join()
            else:
                pipes_results = workers.imap_unordered(self.train_pipe, [x for x in self.gpu_gen(gpu=True)])
                workers.close()
                workers.join()
        else:
            pipes_results = []
            for i, pipe in enumerate(tqdm(self.pipeline_generator(), total=self.gen_len)):
                res = dict(pipe_conf=deepcopy(pipe))
                # start pipeline time
                pipe_start = time()
                if self.cross_validation:
                    cv_score = calc_cv_score(pipe, n_folds=self.k_fold)
                    results = {"test": cv_score}
                else:
                    results = train_evaluate_model_from_config(pipe, to_train=True, to_validate=True)
                res['pipe_time'] = time() - pipe_start
                res['results'] = results

                pipes_results.append(res)

        dataset_res = self.update_logger(pipes_results)

        # save log
        self.logger.log['experiment_info']['full_time'] = normal_time(time() - self.start_exp)
        self.logger.save()

        # delete all checkpoints and save only best pipe
        if self.save_best:
            for name in dataset_res.keys():
                source = join(self.save_path, name)  # , 'pipe_{}'.format(dataset_res[name]["best_ind"])
                dest1 = join(self.save_path, name + '_best_pipe')
                if not os.path.isdir(dest1):
                    os.makedirs(dest1)

                files = os.listdir(source)
                for f in files:
                    if not f.startswith('pipe') and not os.path.isfile(join(dest1, f)):
                        shutil.move(join(source, f), dest1)
                    elif f == 'pipe_{}'.format(dataset_res[name]["best_ind"]):
                        if os.path.isdir(join(dest1, f)):
                            rmtree(join(dest1, f))
                            shutil.move(join(source, f), dest1)
                        else:
                            shutil.move(join(source, f), dest1)

                # del all tmp files in save path
                rmtree(join(self.save_path, name))

        # visualization of results
        path = join(self.root, self.date, self.exp_name)
        results_visualization(path, self.plot, target_metric=self.target_metric)
        return None

    def test(self):
        """
        Initializes the pipeline generator with tiny data and runs the test of experiment.
        """
        # create the pipeline generator
        pipeline_generator = PipeGen(self.exp_config, self.save_path, sample_num=self.sample_num, test_mode=True)
        len_gen = pipeline_generator.length

        # Start generating pipelines configs
        print('[ Test start - {0} pipes, will be run]'.format(len_gen))
        for i, pipe in enumerate(tqdm(pipeline_generator(), total=len_gen)):
            data_iterator_i = self.test_dataset_reader_and_iterator(pipe, i)
            results = train_evaluate_model_from_config(pipe, iterator=data_iterator_i, to_train=True, to_validate=False)
            del results

        # del all tmp files in save path
        rmtree(join(self.save_path, "tmp"))
        print('[ The test was successful ]')
        return None

    def test_dataset_reader_and_iterator(self, config, i):
        # create and test data generator and data iterator
        data = read_data_by_config(config)
        if i == 0:
            for dtype in self.dataset_composition.keys():
                if len(data.get(dtype, [])) != 0:
                    self.dataset_composition[dtype] = True
        else:
            for dtype in self.dataset_composition.keys():
                if len(data.get(dtype, [])) == 0 and self.dataset_composition[dtype]:
                    raise ConfigError("The file structure in the {0} dataset differs "
                                      "from the rest datasets.".format(config['dataset_reader']['data_path']))

        iterator = get_iterator_from_config(config, data)
        if isinstance(iterator, DataFittingIterator):
            raise ConfigError("Instance of a class 'DataFittingIterator' is not supported.")
        else:
            if config.get('train', None):
                if config['train']['test_best'] and len(iterator.data['test']) == 0:
                    raise ConfigError("The 'test' part of dataset is empty, but 'test_best' in train config is 'True'."
                                      " Please check the dataset_iterator config.")

                if (config['train']['validate_best'] or config['train'].get('val_every_n_epochs', False) > 0) and \
                        len(iterator.data['valid']) == 0:
                    raise ConfigError("The 'valid' part of dataset is empty, but 'valid_best' in train config is 'True'"
                                      " or 'val_every_n_epochs' > 0. Please check the dataset_iterator config.")
            else:
                if len(iterator.data['test']) == 0:
                    raise ConfigError("The 'test' part of dataset is empty as a 'train' part of config file, "
                                      "but default value of 'test_best' is 'True'. "
                                      "Please check the dataset_iterator config.")

        # get a tiny data from dataset
        if len(iterator.data['train']) <= 100:
            print("!!!!!!!!!!!!! WARNING !!!!!!!!!!!!! Length of 'train' part dataset <= 100. "
                  "Please check the dataset_iterator config")
            tiny_train = copy(iterator.data['train'])
        else:
            tiny_train = copy(iterator.data['train'][:10])
        iterator.train = tiny_train

        if len(iterator.data['valid']) <= 20:
            tiny_valid = copy(iterator.data['valid'])
        else:
            tiny_valid = copy(iterator.data['valid'][:5])
        iterator.valid = tiny_valid

        if len(iterator.data['test']) <= 20:
            tiny_test = copy(iterator.data['test'])
        else:
            tiny_test = copy(iterator.data['test'][:5])
        iterator.test = tiny_test

        iterator.data = {'train': tiny_train,
                         'valid': tiny_valid,
                         'test': tiny_test,
                         'all': tiny_train + tiny_valid + tiny_test}

        return iterator

    def save_config(self, conf, dataset_name, i) -> None:
        """
        Save train config in checkpoint folder.
        """
        with open(join(self.save_path, dataset_name, "pipe_{}".format(i+1), 'config.json'), 'w') as cf:
            json.dump(conf, cf)
            cf.close()
