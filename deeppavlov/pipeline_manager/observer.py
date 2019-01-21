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


class Observer:
    """
    The :class:`~pipeline_manager.observer.Observer` implements the functions of observing the course of experiments,
    collecting results, time and other useful information, logging and storing it.

    Args:
            name: name of the experiments.
            root: path to root folder.
            info: additional information that you want to add to the log, the content of the dictionary
             does not affect the algorithm
            date: date of the experiment.
    """

    def __init__(self, name: str, root: Union[str, Path], info: dict, date: str, plot: bool) -> None:
        """ Initializes the log, creates a folders tree and files necessary for the observer to work. """

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

        # build folder dependencies
        self.save_path = self.root / self.date / self.exp_name / 'checkpoints'
        self.log_path = self.root / date / self.exp_name
        self.exp_file = self.log_path / (self.exp_name + '.json')
        self.log_file = self.log_path / 'logs.jsonl'

        if not self.save_path.is_dir():
            self.save_path.mkdir(parents=True)
        if self.plot:
            if not (self.log_path / 'images').is_dir():
                (self.log_path / 'images').mkdir()

        self.exp_info = OrderedDict(date=date,
                                    exp_name=self.exp_name,
                                    root=str(self.root),
                                    info=self.exp_inf,
                                    number_of_pipes=None,
                                    metrics=None,
                                    target_metric=None,
                                    dataset_name=None)
        self.log = None

    def save_exp_info(self):
        """ Write exp_info in json file. """
        with open(str(self.exp_file), 'w') as exp_file:
            json.dump(self.exp_info, exp_file)

    def exp_time(self, time: str) -> None:
        """
        Adding the time duration of the experiment in log file.

        Args:
            time: the time duration of the experiment

        Returns:
            None
        """
        with open(str(self.exp_file), 'r') as old_log:
            old_log = json.load(old_log)

        old_log['full_time'] = time
        with open(str(self.exp_file), 'w') as log_file:
            json.dump(old_log, log_file)

    def tmp_reset(self) -> None:
        """ Reinitialize temporary attributes. """
        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None
        self.batch_size = None
        self.log = None

    def write(self) -> None:
        """ Write pipeline logs in jsonl file. """
        if self.log_file.is_file():
            with open(str(self.log_file), 'a') as logs:
                logs.write(json.dumps(self.log))
                logs.write('\n')
        else:
            with open(str(self.log_file), 'w') as logs:
                logs.write(json.dumps(self.log))
                logs.write('\n')

    def update_log(self):
        """ Updates a log with new pipeline information. """

        if (self.model is None) and (self.pipe_conf is not None):
            for component in self.pipe_conf:
                if component.get('main') is True:
                    self.model = component['component_name']

        pipe_name = '-->'.join([x['component_name'] for x in self.pipe_conf])

        self.log = {'pipe_index': self.pipe_ind,
                    'model': self.model,
                    'config': self.pipe_conf,
                    'light_config': pipe_name,
                    'time': self.pipe_time,
                    'batch_size': self.batch_size,
                    'results': self.pipe_res}

        self.write()
        self.tmp_reset()
        return self

    def save_config(self, conf: dict, dataset_name: str, ind: int) -> None:
        """ Save train config in checkpoint folder. """
        with open(str(self.save_path / dataset_name / "pipe_{}".format(ind) / 'config.json'), 'w') as cf:
            json.dump(conf, cf)

    def save_best_pipe(self) -> None:
        """ Calculate the best pipeline and delete others pipelines checkpoints. """
        logs = []
        with open(str(self.log_file), 'r') as log_file:
            for line in log_file.readlines():
                logs.append(json.loads(line))

        with open(self.exp_file, 'r') as info:
            exp_info = json.load(info)

        target_metric = exp_info['target_metric']
        dataset_name = exp_info['dataset_name']

        if 'test' in logs[0]['results'].keys():
            sort_logs = sorted(logs, key=lambda x: x['results']['test'][target_metric], reverse=True)
        else:
            sort_logs = sorted(logs, key=lambda x: x['results']['valid'][target_metric], reverse=True)

        source = self.save_path / dataset_name
        dest1 = self.save_path / (dataset_name + '_best_pipe')
        if not dest1.is_dir():
            dest1.mkdir()

        files = sorted(source.glob("*"))
        for f in files:
            if not f.name.startswith('pipe') and not (dest1 / f.name).is_file():
                shutil.move(str((source / f.name)), str(dest1))
            elif f.name == 'pipe_{}'.format(sort_logs[0]['pipe_index']):
                if (dest1 / f.name).is_dir():
                    rmtree((dest1 / f.name))
                    shutil.move(str(source / f.name), str(dest1))
                else:
                    shutil.move(str(source / f.name), str(dest1))

        # del all tmp files in save path
        rmtree(str(self.save_path / dataset_name))
