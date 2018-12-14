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
import shutil
from collections import OrderedDict
from pathlib import Path
from shutil import rmtree
from typing import Union


class Observer(object):
    """
    The :class:`~pipeline_manager.observer.Observer` implements the functions of observing the course of experiments,
    collecting results, time and other useful information, logging and storing it.
    """

    def __init__(self, name: str, root: Union[str, Path], info: dict, date: str, plot: bool) -> None:
        """
        Initializes the log, creates a folders tree and files necessary for the observer to work.

        Args:
            name: str; name of the experiments.
            root: str; path to root folder.
            info: dict; additional information that you want to add to the log, the content of the dictionary
             does not affect the algorithm
            date: str; date of the experiment.
        """

        self.exp_name = name
        self.exp_inf = info
        self.root = root
        self.date = date
        self.plot = plot

        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None
        self.batch_size = None
        self.dataset = None

        # build folder dependencies
        self.save_path = self.root / self.date / self.exp_name / 'checkpoints'
        self.log_path = self.root / date / self.exp_name
        self.log_file = self.log_path / (self.exp_name + '.json')

        if not self.save_path.is_dir():
            self.save_path.mkdir(parents=True)
        if self.plot:
            if not (self.log_path / 'images').is_dir():
                (self.log_path / 'images').mkdir()

        self.log = OrderedDict(experiment_info=OrderedDict(date=date,
                                                           exp_name=self.exp_name,
                                                           root=str(self.root),
                                                           info=self.exp_inf,
                                                           number_of_pipes=None,
                                                           metrics=None,
                                                           target_metric=None),
                               experiments=OrderedDict())

    def tmp_reset(self) -> None:
        """
        Reinitialize temporary attributes.
        """
        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None
        self.batch_size = None
        self.dataset = None

    def write(self) -> None:
        """
        Write log in log_file.
        """
        if self.log_file.is_file():
            with open(str(self.log_file), 'r') as old_log:
                old_log = json.load(old_log)

            new_log = self.merge_logs(old_log, self.log)
            with open(str(self.log_file), 'w') as log_file:
                json.dump(new_log, log_file)
        else:
            with open(str(self.log_file), 'w') as log_file:
                json.dump(self.log, log_file)

    def exp_time(self, time: str) -> None:
        """
        Adding the time duration of the experiment in log file.

        Args:
            time: the time duration of the experiment

        Returns:
            None
        """
        with open(str(self.log_file), 'r') as old_log:
            old_log = json.load(old_log)

        old_log['experiment_info']['full_time'] = time
        with open(str(self.log_file), 'w') as log_file:
            json.dump(old_log, log_file)

    def update_log(self):
        """ Updates a log with new pipeline information. """

        if (self.model is None) and (self.pipe_conf is not None):
            for component in self.pipe_conf:
                if component.get('main') is True:
                    self.model = component['component_name']

        pipe_name = '-->'.join([x['component_name'] for x in self.pipe_conf])

        if self.dataset not in self.log['experiments'].keys():
            self.log['experiments'][self.dataset] = OrderedDict()

        self.log['experiments'][self.dataset][self.pipe_ind] = {'model': self.model,
                                                                'config': self.pipe_conf,
                                                                'light_config': pipe_name,
                                                                'time': self.pipe_time,
                                                                'batch_size': self.batch_size,
                                                                'results': self.pipe_res}

        self.tmp_reset()
        self.write()
        return self

    def save_config(self, conf: dict, dataset_name: str, ind: int) -> None:
        """ Save train config in checkpoint folder. """
        with open(str(self.save_path / dataset_name / "pipe_{}".format(ind) / 'config.json'), 'w') as cf:
            json.dump(conf, cf)

    def save_best_pipe(self) -> None:
        """ Calculate the best pipeline and delete others pipelines checkpoints. """
        dataset_res = {}

        with open(str(self.log_file), 'r') as log_file:
            log = json.load(log_file)

        target_metric = log['experiment_info']['target_metric']

        for dataset_name in log['experiments'].keys():
            for key, item in log['experiments'][dataset_name].items():
                results = item['results']

                if dataset_name not in dataset_res.keys():
                    dataset_res[dataset_name] = dict(best_score=-1, best_ind=None)

                if 'test' in results.keys():
                    if results['test'][target_metric] > dataset_res[dataset_name]["best_score"]:
                        dataset_res[dataset_name]["best_score"] = results['test'][target_metric]
                        dataset_res[dataset_name]["best_ind"] = key

                else:
                    if results['valid'][target_metric] > dataset_res[dataset_name]["best_score"]:
                        dataset_res[dataset_name]["best_score"] = results['valid'][target_metric]
                        dataset_res[dataset_name]["best_ind"] = key

        for name in dataset_res.keys():
            source = self.save_path / name
            dest1 = self.save_path / (name + '_best_pipe')
            if not dest1.is_dir():
                dest1.mkdir()

            files = sorted(source.glob("*"))
            for f in files:
                if not f.name.startswith('pipe') and not (dest1 / f.name).is_file():
                    shutil.move(str((source / f.name)), str(dest1))
                elif f.name == 'pipe_{}'.format(dataset_res[name]["best_ind"]):
                    if (dest1 / f.name).is_dir():
                        rmtree((dest1 / f.name))
                        shutil.move(str(source / f.name), str(dest1))
                    else:
                        shutil.move(str(source / f.name), str(dest1))

            # del all tmp files in save path
            rmtree(str(self.save_path / name))

    @staticmethod
    def merge_logs(old_log: dict, new_log: dict) -> dict:
        """
        Merge a logs of two experiments.

        Args:
            old_log: config dict
            new_log: config dict

        Returns:
            dict: new config
        """
        n_old = 0
        for dataset_name in old_log['experiments'].keys():
            n_old += len(old_log['experiments'][dataset_name])

        for dataset_name, dataset_val in new_log['experiments'].items():
            if dataset_name not in old_log['experiments'].keys():
                old_log['experiments'][dataset_name] = dataset_val
            else:
                for ind, item in new_log['experiments'][dataset_name].items():
                    if ind not in old_log['experiments'][dataset_name].keys():
                        old_log['experiments'][dataset_name][ind] = item
                    else:
                        for nkey, nval in item.items():
                            match = False
                            for okey, oval in old_log['experiments'][dataset_name][ind].items():
                                if nval['config'] == oval['config']:
                                    match = True
                            if not match:
                                n_old += 1
                                old_log['experiments'][dataset_name][str(n_old)] = nval

        return old_log
